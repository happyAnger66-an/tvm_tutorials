# MLC-LLM 在 NVIDIA 上的实现路径（BYOC / 外挂库）

本文承接教程基线（`simple_llm_decoder.py`：`zero` + DLight + `CodeGenCUDA`）的讨论：  
若算子全靠 TVM 自生成 CUDA kernel，NV 每出新硬件特性，往往要改 codegen / schedule / intrinsic。  
**MLC-LLM 的做法不是否认这条路，而是把热点算子接到厂商/专用库上**，用 TVM 做图编译与长尾算子，降低跟硬件的成本。

> 状态：先落一版总览；后续按章节继续补细节、源码路径与实验笔记。

相关代码树：

- 教程基线：`tvm_tutorials/`
- TVM：`edgeLLM/tvm/`
- MLC-LLM：`edgeLLM/mlc-llm/`

---

## 1. 问题背景：自生成 kernel 与新硬件

### 1.1 TVM 自生成路径（教程基线）

```text
Relax 图
  → LegalizeOps / FuseTIR
  → DLight / MetaSchedule（调度 PrimFunc）
  → CodeGenCUDA（TIR → CUDA C++ 字符串）
  → nvcc / nvrtc → PTX/cubin
```

关键源码（TVM）：

- 调度：`python/tvm/s_tir/dlight/`
- 代码生成：`src/backend/cuda/codegen/codegen_cuda.cc`（入口 `BuildCUDA` / `target.build.cuda`）
- JIT 编译：`python/tvm/support/nvcc.py`（`TVM_CUDA_COMPILE_MODE`）

生成的 `.cu` **不在仓库里**，编译期在内存中生成；可用 `tvm_callback_cuda_postproc` 落盘查看。

### 1.2 跟新硬件的代价

要用上新 ISA / Tensor Core / 新 dtype，通常要动：

| 层次 | 例子 |
|------|------|
| Codegen | 新 intrinsic、PTX、数据类型 |
| Schedule / DLight | 新分块、pipeline、WMMA/MMA 模板 |
| Target | `sm_90` / `sm_100` 等 |

**只换卡、ISA 兼容时**：老 kernel 往往能跑，但不一定吃满新单元。

### 1.3 缓解思路（光谱）

| 策略 | 跟新硬件 | 可控性 |
|------|----------|--------|
| 全 TVM codegen | 社区/自己跟 | 高 |
| TVM + cuBLAS / CUTLASS / FlashInfer | 库侧跟 | 中 |
| TensorRT 等黑盒 | 厂商跟 | 低 |

MLC-LLM 落在中间：**图用 TVM，算力热点 BYOC / extern。**

---

## 2. MLC-LLM 总体策略（总览）

```text
模型图 (Relax)
    │
    ├─ Attention / Paged KV  → FlashInfer（优先）或 TIR 回退
    ├─ 稠密 GEMM             → cuBLAS（pattern BYOC，可关）
    ├─ Hopper / Blackwell    → CUTLASS extern（如 sm_90a / sm_100a）
    ├─ 量化 / MoE 等         → CUTLASS / FasterTransformer / Triton …
    └─ 其余算子              → Legalize → DLight → CodeGenCUDA
            +
         CUDA Graph（decode / verify 等）
```

一句话：

- **tutorials** = 证明 TVM CUDA codegen 能跑通 decoder；
- **mlc-llm** = 同一底座上接 FlashInfer / cuBLAS / CUTLASS + PagedKV + CUDA Graph，做成可部署引擎。

---

## 3. 优化开关与预设

入口：`mlc-llm/python/mlc_llm/interface/compiler_flags.py`（`OptimizationFlags`）。

| 标志 | 作用 |
|------|------|
| `flashinfer` | Paged KV attention 等走 FlashInfer |
| `cublas_gemm` | 稠密 GEMM 图级 pattern → cuBLAS |
| `cutlass` | 模型侧 `cutlass.*` extern（受 arch 校正） |
| `cudagraph` | `RewriteCUDAGraph` 等 |
| `faster_transformer` | FT 量化 GEMM 等 |

预设（摘要）：

| 预设 | 大致打开 |
|------|----------|
| O0 | 全关 |
| O1 | cuBLAS + CUTLASS + FT |
| **O2** | FlashInfer + cuBLAS + CUTLASS + CUDA Graph（常用） |
| O3 | O2 + 更多（如 FT、IPC allreduce 等） |

`update(target, quantization)` 会校正：例如 FlashInfer 要求 CUDA 且 arch ≥ 80；CUTLASS 是否真正启用与 arch 相关（见 `op/extern.py`）。

全局外挂开关：`mlc-llm/python/mlc_llm/op/extern.py` 的 `enable(...)`。

---

## 4. 编译流水线如何分流

注册管线：`mlc-llm/python/mlc_llm/compiler_pass/pipeline.py`（`@register_pipeline("mlc_llm")`）。

### 4.1 阶段概览（后续细拆）

1. **KV Cache 物化** — `DispatchKVCacheCreation`  
   - 总是生成 `create_tir_paged_kv_cache`  
   - 条件满足再生成 `create_flashinfer_paged_kv_cache`，FlashInfer JIT 产物进 `extern_mods`
2. **图级库分发（Legalize 前）** — 如 `BLASDispatch`（cuBLAS）、Triton / FT 相关 pass  
3. **零管线同类步骤 + DLight** — `LegalizeOps` → fuse → `ApplyDefaultSchedule`  
4. **收尾** — 内存规划、`RewriteCUDAGraph`、`AttachExternModules` 等

### 4.2 各后端接入方式（要点）

| 后端 | 接入形态 | 说明 |
|------|----------|------|
| FlashInfer | JIT 外部模块 + runtime 优先创建 FI KV | 主路径在 attention / paged KV，不是改 `codegen_cuda` |
| cuBLAS | `FuseOpsByPattern` + `RunCodegen`（BYOC） | 图上抠 GEMM 子图 |
| CUTLASS | 多在模型/量化里 `op.extern("cutlass.*")` | **非** `partition_for_cutlass`；详解见 §10 |
| DLight + CodeGenCUDA | 未外挂的长尾算子 | 与教程同族 |

> TODO：补 `blas_dispatch.py` / `dispatch_kv_cache_creation.py` 的调用顺序与 pattern 列表。

---

## 5. 运行时行为（摘要）

- Serve 创建 KV（如 `cpp/serve/function_table.cc`）：**优先 FlashInfer，否则 TIR**。
- 统一协议仍接近 `vm.builtin.kv_state_*` / attention KV cache builtins。
- CUDA Graph：编译期 `relax.backend.use_cuda_graph` + `RewriteCUDAGraph`；加载时可跑 alloc init。
- Sampling：有 FlashInfer sampling 符号时优先走库。

> TODO：补 batch prefill/decode/verify 与 capture 范围。

---

## 6. 构建与依赖（NVIDIA）

常见 CMake / 文档推荐（见 mlc `docs/install/tvm.rst`、`cmake/gen_cmake_config.py`）：

- `USE_CUDA ON`
- `USE_CUBLAS ON`
- `USE_CUTLASS ON`
- `USE_THRUST ON`

FlashInfer：当前主路径多依赖 **`flashinfer-python`**，由 TVM `relax.backend.cuda.flashinfer` 做 JIT；与历史 `USE_FLASHINFER` CMake 选项可能并存，以现网路径为准。

多 arch：`CMAKE_CUDA_ARCHITECTURES` / `MLC_MULTI_ARCH` 等（fatbin）。

---

## 7. 与 `tvm_tutorials` 基线对比

| | `simple_llm_decoder.py` | MLC-LLM |
|--|--|--|
| Attention | 手写 matmul + mask + softmax | PagedKV + FlashInfer / TIR |
| KV cache | 无 | 有（FI 优先） |
| 编译 | `zero` + DLight | `mlc_llm` pipeline（外包一层再 DLight） |
| GEMM 库 | 无（纯 codegen） | 可选 cuBLAS / CUTLASS / FT / Triton |
| CUDA Graph | 无（compile 默认管线可能带 rewrite，教程未作为主路径） | 显式开关 + serve 集成 |
| 目标 | 教学 / 对照 diff | 生产级编译与 serve |

---

## 8. 和「要不要改 codegen_cuda」的关系

| 算子 | 新卡特性主要由谁跟 |
|------|-------------------|
| Attention / Paged KV | FlashInfer（及 TVM 绑定层） |
| 大 GEMM | cuBLAS / CUTLASS |
| Norm、小算子、未外挂部分 | 仍是 DLight + `codegen_cuda` |

结论：MLC-LLM **没有取消** TVM 自生成路径，而是把最吃新硬件的部分 **BYOC / extern 外包**；长尾仍要接受 codegen 演进成本。

---

## 9. 关键路径速查

**MLC-LLM**

- `python/mlc_llm/interface/compile.py` — 编译入口、`op_ext.enable`、pass config
- `python/mlc_llm/interface/compiler_flags.py` — O0–O3 / 各 flag
- `python/mlc_llm/compiler_pass/pipeline.py` — `mlc_llm` 流水线
- `python/mlc_llm/compiler_pass/blas_dispatch.py` — cuBLAS
- `python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py` — KV 后端
- `python/mlc_llm/op/extern.py` — 外挂全局开关
- `cpp/serve/function_table.cc` — 运行时选 FI / TIR KV

**TVM（被 mlc 调用）**

- `python/tvm/relax/frontend/nn/llm/kv_cache.py` — `FlashInferPagedKVCache` / `TIRPagedKVCache`
- `python/tvm/relax/backend/cuda/flashinfer.py`
- `python/tvm/relax/backend/cuda/cublas.py`
- `python/tvm/relax/backend/cuda/cutlass.py`
- `src/backend/cuda/codegen/codegen_cuda.cc`

---

## 10. CUTLASS extern 在量化模型里怎么插（详解）

和 cuBLAS 的 **图级 pattern BYOC**（`FuseOpsByPattern`）不同：MLC 的 CUTLASS 主路径是 **建图时显式插入** `nn.op.extern("cutlass.*", ...)`，导出成 Relax 的 `call_dps_packed("cutlass.xxx", ...)`，运行时直接调 TVM 里用 CUTLASS 注册的全局函数。  
**不是** pipeline 里跑 `partition_for_cutlass` 自动抠子图。

### 10.1 端到端时间线

```text
compile.py
  └─ op_ext.enable(..., cutlass=opt.cutlass)
        └─ 仅 cuda 且 arch ∈ {sm_90a, sm_100a} 时
           STORE.cutlass_gemm / cutlass_group_gemm = True
  └─ quantization.quantize_model(model)
        └─ Mutator：nn.Linear → BlockScaleQuantizeLinear / PerTensor… 等
  └─ model.export_tvm / 跑 forward 建图
        └─ 量化 Linear.forward 内：
              if STORE.cutlass_* and runtime 已注册 cutlass.*:
                  cutlass.fp8_* / group_gemm  → op.extern("cutlass.xxx")
              else:
                  Triton / 普通 matmul / moe_matmul 回退
  └─ IR 里已是 call_dps_packed("cutlass.xxx")
        └─ 后续 Legalize/DLight 不会再把它变成 TIR matmul
  └─ 运行时：TVM contrib CUTLASS 内核（.cu 里 GlobalDef 注册的符号）
```

关键文件：

| 环节 | 路径 |
|------|------|
| 打开开关 | `mlc_llm/op/extern.py` → `enable()` |
| 编译入口调用 | `mlc_llm/interface/compile.py` |
| 包装 `op.extern` | `mlc_llm/op/cutlass.py` |
| 量化层插入点 | `quantization/block_scale_quantization.py`、`per_tensor_quantization.py`、`fp8_quantization.py` |
| `extern` → IR | TVM `relax/frontend/nn/op.py` → `call_dps_packed` |
| 运行时实现 | TVM `src/runtime/extra/contrib/cutlass/*.cu`（如 `fp8_groupwise_scaled_gemm_sm90.cu`） |

### 10.2 开关：何时允许插 CUTLASS

`op/extern.py`：

```python
cutlass = (
    cutlass
    and target.kind.name == "cuda"
    and target.attrs.get("arch", "") in ["sm_90a", "sm_100a"]  # Hopper / Blackwell
)
STORE.cutlass_gemm = cutlass
STORE.cutlass_group_gemm = cutlass
```

含义：

- CLI / O2 里 `cutlass=1` 只是**意愿**；
- 真正插 extern 还要求 **Hopper(`sm_90a`) 或 Blackwell(`sm_100a`)**；
- 普通 Ampere（如 `sm_80`）即使开了 flag，STORE 仍为 False，量化 forward 会走 Triton / 普通路径。

### 10.3 先换层，再在 forward 里决定算子

以 **block-scale FP8**（DeepSeek 等常用）为例：

1. `BlockScaleQuantize.quantize_model` 用 `nn.Mutator` 遍历模型；
2. 遇到普通 `nn.Linear`（非 final FC、非 MoE gate）→ 换成 `BlockScaleQuantizeLinear`（或带 static activation scale 的变体）；
3. MoE experts → `BlockScaleQuantizeMixtralExperts`；
4. 权重变成 `float8` + `weight_scale_inv`（block scale），**此时还没调用 CUTLASS**；
5. 真正插入发生在后续 **建图时执行 `forward`**：里面读 `extern.get_store().cutlass_gemm`，再决定 `cutlass.*` 还是 Triton。

也就是说：**量化改的是 Module 类型与参数布局；CUTLASS 是 forward 里的条件分发。**

### 10.4 插入点 A：普通 Linear（block-scale FP8）

`BlockScaleQuantizeLinear.forward`（`block_scale_quantization.py`）逻辑概要：

```text
if m == 1:                          # 单 token / GEMV
    → TIR dequantize GEMV（不走 CUTLASS）

else if cutlass_gemm
     and 运行时存在 "cutlass.groupwise_scaled_gemm_e4m3fn_e4m3fn":
    → rowwise_group_quant_fp8(x) 得到 x_fp8, x_scale
    → cutlass.fp8_groupwise_scaled_gemm(...)

else:
    → 同样量化 activation，但走 triton.fp8_groupwise_scaled_gemm
```

`cutlass.fp8_groupwise_scaled_gemm`（`op/cutlass.py`）内部：

```python
workspace = op.empty((4096 * 1024,), dtype="uint8")
return op.extern(
    "cutlass.groupwise_scaled_gemm_e4m3fn_e4m3fn",
    args=[x, weight, x_scale, weight_scale, workspace, block_h, block_w],
    out=Tensor.placeholder((..., n), dtype=out_dtype),
)
```

约束（代码里写死/检查）：

- weight / act 为 `float8_e4m3fn`，scale 为 `float32`；
- `block_size == (128, 128)`；
- 输出 `float16` / `bfloat16`；
- 分配一块 workspace buffer 给 CUTLASS。

Static activation 变体（`BlockScaleQuantizeLinearStaticActivation`）类似，但 activation scale 用标定好的静态 scale；并额外检查 `weight` 维能被 128 整除。

### 10.5 插入点 B：Per-tensor FP8 Linear

`PerTensorQuantizeLinear.forward`（`per_tensor_quantization.py`）：

- `calibration_mode == "inference"` 且权重已是 FP8 storage 时：
  - 若 `cutlass_gemm` 且 **非单元素 batch**（`prod(shape[:-1]) != 1`）→ `cutlass.fp8_gemm` → 符号如 `cutlass.gemm_e4m3_e4m3_fp16`；
  - 否则 → `nn.op.matmul` + scale（可被后续 cuBLAS pattern 接住，那是另一条路）。

单 token decode（`m==1`）刻意避开 CUTLASS gemm，偏 GEMV / 普通 matmul。

### 10.6 插入点 C：MoE group GEMM

**Per-tensor FP8 MoE**（`fp8_quantization.py` → `FP8PerTensorQuantizeMixtralExperts`）：

```text
if indptr.ndim == 2:          # 单 token 特化
    → moe_matmul.dequantize_float8_gemv
elif cutlass_group_gemm:
    → cutlass.group_gemm(...)   # cutlass.group_gemm_e4m3_e4m3_fp16 等
else:
    → dequantize + moe_matmul.group_gemm（TIR 回退）
```

**Block-scale FP8 MoE**（`BlockScaleQuantizeMixtralExperts.forward`）：

```text
if indptr.ndim == 2:
    → TIR block-scale GEMV
elif cutlass_gemm and 存在 groupwise_scaled_group_gemm 符号:
    → cutlass.fp8_groupwise_scaled_group_gemm
else:
    → triton.fp8_groupwise_scaled_group_gemm
```

### 10.7 `op.extern`：使用场景、原理与实现流程

以 `mlc_llm/op/cutlass.py` 中 `fp8_groupwise_scaled_gemm` 为例：

```python
func_name = "cutlass.groupwise_scaled_gemm_e4m3fn_e4m3fn"
workspace = op.empty((4096 * 1024,), dtype="uint8", name="workspace")
return op.extern(
    func_name,
    args=[x, weight, x_scale, weight_scale, workspace, block_size[0], block_size[1]],
    out=nn.Tensor.placeholder((*x.shape[:-1], weight.shape[0]), dtype=out_dtype),
)
```

含义：不让 TVM `Legalize → DLight → CodeGenCUDA` 生成 GEMM，而是在图里**点名**调用已注册的运行时 packed 函数。

#### 10.7.1 使用场景

| 场景 | 例子 |
|------|------|
| 厂商 / 专用库 kernel | CUTLASS FP8 GEMM、部分 FlashInfer 路径 |
| 手写 CUDA/C++ 并 `GlobalDef` 注册 | 自定义 attention、量化 kernel |
| 避开自生成 matmul 路径 | 性能或新硬件特性已在库侧实现 |
| 需要 workspace / 特殊 dtype / 非标准 epilogue | 如 CUTLASS 的 workspace buffer |

**不适合**：普通可融合小算子（应用 `op.add` / `op.matmul`，交给编译器）；未注册或签名不对的函数（运行期才失败）。

与 **cuBLAS BYOC** 的差别：BYOC 由 pass 从 `matmul` 图 pattern 抠出；`extern` 是**建图时手写插入**。

#### 10.7.2 原理：Destination-Passing Style（DPS）

`op.extern` 底层是 `relax.call_dps_packed`（定义见 TVM `python/tvm/relax/op/base.py`、前端包装见 `relax/frontend/nn/op.py`）：

1. 调用方**先按 `out` 的 StructInfo 分配输出 buffer**；
2. 把「输入 + 输出」一并传给 packed func；
3. 外部函数**就地写入 out**，不靠返回值新建张量；
4. 编译器假定除写指定 out 外函数**纯净**（可重排/消除）；有其它副作用则危险。

语义等价于：

```text
out = alloc_tensor(shape, dtype)
cutlass.xxx(x, w, scales..., workspace, block_h, block_w, out)
return out
```

#### 10.7.3 实现流程（从 `op.extern` 到 GPU）

**步骤 1 — 前端 `nn.op.extern`**

1. 把 `args` 转成 Relax 表达式：Tensor→Expr，`int`→`PrimValue`，`str`→`StringImm` 等；  
2. 从 `out` placeholder 取出 `TensorStructInfo`（声明输出 shape/dtype）；  
3. 生成 `call_dps_packed(ExternFunc(name), args, out_sinfo)`，再 `wrap_nested` 成 `nn.Tensor`。

注意：`out=` **不是**运行时已有的真实 buffer，而是**类型/形状声明**；真分配在后续 rewrite。

**步骤 2 — export / 建图**

量化 `Linear.forward` 调到此处时，BlockBuilder 记下类似节点：

```text
lv = R.call_dps_packed(
    "cutlass.groupwise_scaled_gemm_e4m3fn_e4m3fn",
    (x, w, x_scale, w_scale, workspace, 128, 128),
    out_sinfo=R.Tensor((m, n), out_dtype)
)
```

`LegalizeOps` **不会**把它拆成 TIR matmul；符号名保留到 runtime。

**步骤 3 — `CallTIRRewrite`（lowering）**

对 `call_dps_packed` / `call_tir` 统一改写（`src/relax/transform/call_tir_rewrite.cc`）：

```text
alloc = relax.builtin.alloc_tensor(shape, dtype, device)
call_packed(func, *inputs, alloc)   # 输出作为末尾参数
```

DPS 在 IR 上变成「先 alloc，再 packed 调用」。

**步骤 4 — 运行时按名查找**

TVM 在 `USE_CUTLASS` 构建下注册全局符号，例如：

- `src/runtime/extra/contrib/cutlass/fp8_groupwise_scaled_gemm_sm90.cu`  
  → `.def("cutlass.groupwise_scaled_gemm_e4m3fn_e4m3fn", ...)`
- `fp8_gemm.cu` → `cutlass.gemm_e4m3_e4m3_fp16` 等

VM 执行到该节点 → `GetGlobalFunc(name)` → 调 CUTLASS 实现，写入事先 alloc 的 `out`（以及传入的 `workspace`）。

建图前 mlc 还会 `tvm.get_global_func("cutlass.xxx", allow_missing=True)` **探测**宿主是否编进该符号；没有则即使 `STORE.cutlass_*` 打开也回退 Triton/TIR。

#### 10.7.4 结合 `fp8_groupwise_scaled_gemm` 的参数

| 参数 | 作用 |
|------|------|
| `x`, `weight` | FP8 激活 / 权重 |
| `x_scale`, `weight_scale` | block-wise scale |
| `workspace` | CUTLASS 临时区（`op.empty` 进图，运行时分配） |
| `block_size[0/1]` | 编译期常量 → `PrimValue` |
| `out=placeholder(...)` | 声明结果 `[..., n]` 与 dtype，供 alloc + 类型推导 |

#### 10.7.5 流程串起来

```text
cutlass.fp8_groupwise_scaled_gemm(...)
  → op.extern(name, args, out_sinfo)
  → R.call_dps_packed(ExternFunc(name), args)      # export 后
  → alloc_tensor + call_packed(name, args..., out) # CallTIRRewrite
  → VM: GlobalFunc("cutlass.xxx")(DLTensors...)     # 真 CUTLASS kernel
```

**一句话**：`op.extern` = 在 nn 图里声明「调用已注册的 DPS packed 函数」；编译器按 `out` 的 sinfo 分配输出，**不负责**生成该算子的 CUDA；实现必须事先用同名 `GlobalDef` / `register_global_func` 挂进 runtime。

### 10.8 和 `partition_for_cutlass` / cuBLAS 的区别

| | CUTLASS extern（本节） | cuBLAS BYOC | TVM `partition_for_cutlass` |
|--|--|--|--|
| 谁决定 | 量化 `Linear.forward` 手写分支 | pipeline `BLASDispatch` | 图 pattern（mlc 主路径未用） |
| IR 形态 | 建图即 `call_dps_packed("cutlass.*")` | fuse 后 `relax.ext.cublas` | composite → cutlass codegen |
| 典型负载 | FP8 block-scale / group GEMM / MoE | 稠密 fp16/bf16/… GEMM | 通用 matmul+epilogue 等 |
| 硬件门槛 | 基本仅 sm_90a / sm_100a | 更广的 CUDA | 取决于 CUTLASS 构建 |

### 10.9 小结（量化里“怎么插”一句话）

1. **编译前**按 target 打开 `STORE.cutlass_*`；  
2. **量化 Mutator** 把 `Linear`/MoE 换成带 FP8 权重与 scale 的模块；  
3. **export / forward** 时按「开关 + arch + 形状 + 运行时是否注册」选择 `cutlass.fp8_*` / `group_gemm`；  
4. 包装层统一 `op.extern` → `call_dps_packed`；  
5. 跑起来调 TVM 链进来的 CUTLASS `.cu` 实现，失败条件则 Triton/TIR。

---

## 11. 后续待完善章节（占位）

- [ ] FlashInfer：JIT 模块生成、与 `attention_with_fused_qkv` 的绑定、dtype/RoPE 限制  
- [ ] cuBLAS BYOC：pattern 表、哪些 entry function 会 partition、与单 batch decode 的边界  
- [x] CUTLASS：`op.extern` 插入点、量化/MoE/group GEMM、与 `partition_for_cutlass` 的差异 → 见 §10  
- [x] `op.extern` 使用场景 / DPS 原理 / 实现流程 → 见 §10.7  
- [ ] CUDA Graph：capture 函数集合、alloc init、和 serve 的交互  
- [ ] Triton / FT 回退路径（与 §10 回退分支交叉）  
- [ ] Jetson / 端侧：arch、库可用性、与桌面数据中心的差异  
- [ ] 对照实验：O0 vs O2 算子落点（dump IR / `get_source` / profiler）

---

## 12. 参考

- 本仓库：`simple_llm_decoder.py`、`paged_kv_cache/README.md`、`docs/custom_tvm.md`
- MLC-LLM：`docs/install/tvm.rst`、`python/mlc_llm/compiler_pass/pipeline.py`、`python/mlc_llm/op/cutlass.py`、`python/mlc_llm/quantization/block_scale_quantization.py`
- TVM：`python/tvm/relax/frontend/nn/op.py`（`extern`）、`python/tvm/relax/op/base.py`（`call_dps_packed`）、`src/relax/transform/call_tir_rewrite.cc`、`src/runtime/extra/contrib/cutlass/`
