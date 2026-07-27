# TVM BYOC：现有后端、pi05→TensorRT 实践，与扩展新后端

本文整理当前 `edgeLLM/tvm` 的 BYOC 版图、pi05 整图 offload 到 TensorRT 的工作与代价，
以及**通常如何使用 / 如何扩展**一个新后端。可与下列文档对照阅读：

| 文档 | 内容 |
|------|------|
| [`../mlc_llm_byoc.md`](../mlc_llm_byoc.md) | MLC-LLM 在 NVIDIA 上挂 FlashInfer / cuBLAS / CUTLASS |
| [`../custom_tvm.md`](../custom_tvm.md) | 开放 IR 编译框架 vs 黑盒引擎选型 |
| `mlc-vla/docs/byoc/offload_trt.md` | pi05 → TensorRT 落地记录（实测与踩坑） |
| `mlc-vla/docs/byoc/fp8_tvm_trt.md` / `trt_quantization.md` | TRT 路径上 FP8 / NVFP4 方案 |

相关代码树：

- TVM：`/home/zhangxa/codes/edgeLLM/tvm`
- 教程：`tvm_tutorials/`（本仓库）
- pi05 TRT 应用：`mlc-vla/`

---

## 1. 先分清：BYOC 决策发生在哪

**不是**在「tir → 后端 build」时对每个 PrimFunc 再投票选 CodeGenCUDA 或 BYOC。  
分流在更早的 **Relax 图阶段** 完成；build 只按已标注属性各走各路。

| 路径 | 怎么被选中 | build 时走谁 |
|------|------------|--------------|
| TVM 自生成 | 普通 `R.call_tir(PrimFunc, …)`，无外挂标注 | `target.build.cuda` → CodeGenCUDA |
| Pattern BYOC（如 cuBLAS） | `FuseOpsByPattern` 匹配成功，函数打上 `"Codegen": "cublas"` | `RunCodegen` → 外挂后端 |
| 显式 extern（如 MLC 的 CUTLASS） | 建图时就是 `call_dps_packed("cutlass.xxx", …)` | 直接调已注册全局函数 |
| 对象级选择（FlashInfer KV） | 创建 `FlashInferPagedKVCache` 而非 `TIRPagedKVCache` | FI JIT 模块 + runtime |

时间线：

```text
Relax 高级图
    │
    ├─① 可选：FuseOpsByPattern(cuBLAS / TensorRT …)   ← 决定「这块外挂」
    ├─①' 或：模型里已写 op.extern / FI KV              ← 建图时就决定了
    │
    ├─② Legalize / Fuse / DLight                       ← 只处理仍留在 TVM 的部分
    │
    └─③ tvm.compile / build
         ├─ 有 Codegen 属性的函数 → RunCodegen → BYOC
         ├─ call_dps_packed("cutlass.*") → 已注册 runtime 符号
         └─ 其余 PrimFunc → target.build.cuda → CodeGenCUDA
```

教程基线（`simple_llm_decoder.py` / `paged_kv_cache/`）默认**全走 CodeGenCUDA**；
MLC-LLM / pi05-TRT 才会在 Legalize 前插入分区 pass。

---

## 2. 当前 TVM 支持哪些 BYOC 后端

依据 `python/tvm/relax/backend/` 与 `cmake/config.cmake` 中的开关，按平台归类如下。

### 2.1 NVIDIA / CUDA 族

| 后端 | 源码位置 | 典型用途 | CMake / 接入形态 |
|------|----------|----------|------------------|
| **cuBLAS** | `relax/backend/cuda/cublas.py` | 稠密 GEMM | `USE_CUBLAS`；pattern BYOC |
| **cuDNN** | `relax/backend/cuda/cudnn.py` | conv / attention 等 | `USE_CUDNN`；pattern BYOC |
| **CUTLASS** | `relax/backend/cuda/cutlass.py` + `3rdparty/` | GEMM / attention / 量化 gemm | `USE_CUTLASS`；pattern 或 `extern("cutlass.*")` |
| **FlashInfer** | `relax/backend/cuda/flashinfer.py` | Paged KV attention | `USE_FLASHINFER` / python JIT；偏 extern 模块 |
| **TensorRT** | `relax/backend/contrib/tensorrt.py` + `src/runtime/extra/contrib/tensorrt/` + `src/relax/backend/contrib/tensorrt/` | **整段子图** offload 成 TRT engine | `USE_TENSORRT_CODEGEN` + `USE_TENSORRT_RUNTIME` |

### 2.2 其他平台

| 后端 | 源码位置 | 典型用途 | 备注 |
|------|----------|----------|------|
| **hipBLAS** | `relax/backend/rocm/hipblas.py` | ROCm 上 GEMM | pattern BYOC |
| **CLML** | `relax/backend/adreno/clml.py` | 高通 Adreno | `USE_CLML` |
| **NNAPI** | `relax/backend/contrib/nnapi.py` | Android NNAPI | `USE_NNAPI_CODEGEN` / `USE_NNAPI_RUNTIME` |
| **CoreML** | `relax/backend/metal/coreml.py` | Apple | `USE_COREML` |
| **DNNL** | cmake `USE_DNNL` | CPU 库 | 偏经典 contrib 路径 |
| **example_npu** | `relax/backend/contrib/example_npu/` | 教学用假 NPU | **扩展新后端的官方示范** |

### 2.2 「类 BYOC」但不走 classic `Codegen=` 分区

- **`nn.op.extern` / `call_dps_packed`**：MLC 里 CUTLASS 主路径常用
- **专用 runtime 对象**：`FlashInferPagedKVCache` vs `TIRPagedKVCache`

CMake 默认多为 `OFF`，本机是否启用取决于 `build/config.cmake`。

---

## 3. pi05 BYOC 到 TensorRT：做了哪些工作

> 落地摘要见 `mlc-vla/docs/byoc/offload_trt.md`（约 2026-07-25）。  
> TVM 侧提交参考：`a2b3e1a3f feat: support BYOC tensorrt.`；mlc-vla：`712be82 feat: add trt BYOC`。

### 3.1 目标与结果

把 pi05 的 Relax 图（PaliGemma prefill + flow-matching denoise）在 **Legalize 之前** 分区交给 TensorRT：

| 路径 | 稳态延迟（Ada / LIBERO 设定） | 数值 |
|------|-------------------------------|------|
| 纯 TVM（dlight + cuBLAS） | ~187 ms | cosine ≈ 0.9988 |
| TVM → TensorRT（权重固化） | **~165 ms** | 数值对齐 |
| TRT 但权重当 network input | ~213 ms（反而更慢） | — |

约九成算子进 TRT，切成 **11 个 region**（非单 engine）。首次 build engine 约数分钟，缓存后启动约 0.4 s。

### 3.2 TVM 侧改动（必须改内核）

| 位置 | 工作 |
|------|------|
| `python/tvm/relax/backend/contrib/tensorrt.py` | Relax pattern 表 + `partition_for_tensorrt` |
| `tensorrt_ops.cc` | 新增 `matmul` / `gelu` / `silu` / `square` / `astype` / `broadcast_to` 等 converter；修 reshape/reduce 属性读法 |
| `tensorrt_builder.cc` | **逐层精度约束** + 自动 `kFP16`（缺了会在真模型上炸 RMSNorm） |
| `tensorrt_runtime.cc` | engine 缓存 key 混入 graph JSON 哈希 |
| `codegen.cc` / `merge_composite_functions.cc` / `fuse_ops.cc` / `block_builder.cc` | 多 entry、tuple 输出、输入顺序、常量去重等 |

pattern 覆盖：elementwise、激活、shape/layout、reduce、N-D `matmul`。  
主要缺口：`relax.split`（composite 返回 tuple，跨区域边界时表示困难）——留给 TVM 反而区域更少（11 vs 13）。

### 3.3 应用侧（mlc-vla）改动

编译顺序（**顺序不能乱**）：

```text
build_irmodule（param_mode="packed"）
   │  bind_packed_params：packed_params[i] → relax.const，并摘掉该形参
   ▼
apply_tensorrt_prepasses
   partition_for_tensorrt → RunCodegen → DeadCodeElimination
   ▼
apply_gemm_prepasses（cuBLAS，吃掉 TRT 未覆盖的残留 matmul）
   ▼
get_default_pipeline（LegalizeOps + dlight）
```

关键工程点：

1. **权重必须固化进 engine**  
   - 只把绑定右值换成常量不够：`FuseOpsByPattern(bind_constants=True)` 要求算子实参是直接的 `Constant`  
   - 形参必须摘掉，否则 VM 仍要上传整份权重（12GB 卡易 OOM）  
   - 做错 → engine ~4MB、213ms；做对 → engine ~4.5GB、165ms
2. **engine 缓存按 checkpoint tag 分目录**（缓存 key 看不到权重值）
3. 开 TRT 时强制关 CUDA Graph（TRT extern 不在可捕获集合内）
4. runner / 评测配置 / `compare_trt.py` 适配新的调用约定（`weights_bound` 时少传 params）

### 3.4 后续（未全部落地）

- FP8 / NVFP4：见 `fp8_tvm_trt.md`、`trt_quantization.md`（需 Q/DQ converter、strongly-typed network、ModelOpt scale 导出等）

---

## 4. 为什么「整图 TRT」工作量这么大？

库级 BYOC（cuBLAS）与引擎级 BYOC（TensorRT）不是一个量级：

| 难点 | 说明 |
|------|------|
| 语义鸿沟 | Relax op ≠ TRT layer；每个算子要写 converter，名字还要对齐历史 Relay key |
| 分区碎片 | 不能 offload 的 op（如 `split`）撕碎图；边界 tuple/嵌套输出要动 fuse/merge 基建 |
| 精度陷阱 | TRT10 的 dtype 约束与 TVM 声明不一致；小模型对拍过、真模型可能炸——必须改 builder |
| 权重契约 | TRT 要吃 `nvinfer1::Weights` 才能做布局/融合；`packed_params` 需自研绑定 pass |
| 与其它 BYOC 抢图 | TRT 与 cuBLAS 都抢 `matmul`，必须严格排序且都在 Legalize 前 |
| 工程配套 | engine 缓存、显存、与 CUDA Graph 互斥、真权重验收 |

一句话：

- **库级 BYOC**：抠子图 → 调库 API，改 pattern + 少量 runtime 即可  
- **引擎级 BYOC**：等于再接一套图编译器 + runtime + 精度/权重契约（pi05→TRT 属于后者）

---

## 5. 通常使用场景

| 场景 | 典型后端 | 动机 |
|------|----------|------|
| 吃满 vendor 热点算子 | cuBLAS / cuDNN / CUTLASS / FlashInfer | 少写 schedule，跟新硬件 |
| 整网 / 大块子图交给厂商引擎 | TensorRT / CoreML / NNAPI / CLML | 融合、tactic、量化、部署生态 |
| 自研 NPU | example_npu 模式 / 自写 Codegen | 自定义 ISA；长尾仍回 TVM |
| 教程 / 研究可控路径 | 纯 CodeGenCUDA + DLight | 可改 IR、可对照 MLC |

选型经验：

- **只想加速 GEMM / Attention**：优先 cuBLAS / FlashInfer / CUTLASS extern，别上完整 TRT  
- **要整图吃 TRT**：接受改 TVM C++ + 权重绑定 + 真模型验收  
- **跨 Jetson ↔ 非 NVIDIA**：TRT 帮不上忙，仍要 TVM 自生成或其它厂商 BYOC

---

## 6. 扩展新后端的三种方式（由轻到重）

### A. 显式 extern（最轻）

```text
模型里：op.extern("my_lib.gemm", ...)
导出：call_dps_packed("my_lib.gemm", ...)
运行时：TVM 全局函数 / 动态库符号
```

适合：少量固定 kernel、接口稳定（MLC CUTLASS 主路径类似）。

### B. Pattern BYOC（中等，最常见）

```text
1. register_patterns([("foo.matmul", pattern, check), ...])
2. partition_for_foo = FuseOpsByPattern(..., annotate_codegen=True)
3. RunCodegen → 按 Codegen="foo" 分发给后端
4. C++/Python runtime 调库（cuBLAS handle 等）
```

参考：

- `python/tvm/relax/backend/cuda/cublas.py` → `partition_for_cublas`
- pattern 注册：`relax/backend/pattern_registry.py`

适合：从图中自动抠出可加速子图。

### C. 引擎级 BYOC（最重）

```text
1. 尽量广的 pattern 表（覆盖 elementwise / matmul / …）
2. FuseOpsByPattern + MergeCompositeFunctions → 大 region
3. Codegen：Relax 子图 → 厂商 IR / JSON
4. Builder：编译成 engine（含精度策略、权重绑定）
5. Runtime：加载 / 执行 / 缓存 engine
6. 应用侧：Legalize 前插入分区；处理多 region、tuple、与其它 BYOC 排序
```

参考：

- TensorRT：`contrib/tensorrt.py` + `src/runtime/extra/contrib/tensorrt/` + `src/relax/backend/contrib/tensorrt/codegen.cc`
- 教学骨架：`contrib/example_npu/`

适合：TensorRT / CoreML / NNAPI / 自研整图引擎。

### 6.1 扩展检查清单（建议顺序）

1. **定目标**：库级加速 vs 整图引擎？覆盖哪些 Relax op？切不进去是否允许回退 TVM？  
2. **写 pattern**：`register_patterns`，名字与 converter key 一致（如 `"tensorrt.transpose"`）  
3. **写 partition API**：`partition_for_xxx`，明确 `annotate_codegen` / `MergeCompositeFunctions`  
4. **实现 codegen / runtime**（引擎级才需要完整 builder）  
5. **CMake 开关**：`USE_XXX_CODEGEN` / `USE_XXX_RUNTIME` 可分离探测  
6. **插入 pipeline**：必须在 `LegalizeOps` 之前；与其它 BYOC 的抢图顺序写清楚  
7. **验收**：单算子 → dummy 小模型 → **真权重**端到端（精度坑常只在大 shape 暴露）

---

## 7. 和本教程仓库的关系

```text
tvm_tutorials/
  simple_llm_decoder.py          ← 纯 CodeGenCUDA（学通编译栈）
  paged_kv_cache/                ← TIRPagedKVCache，仍是自生成 TIR
  docs/mlc_llm_byoc.md           ← MLC 如何挂库（路径 B / FlashInfer）
  docs/byoc/byoc_new_backend.md  ← 本文：后端清单 + TRT 实践 + 如何扩展
```

学习建议：

1. 先用 `learn_compare.py stages` 看清 Legalize / Fuse / DLight / codegen 自生成路径  
2. 再读 `mlc_llm_byoc.md`，理解「热点 BYOC、长尾 codegen」  
3. 若要做新 NPU / 新引擎，从 `example_npu` 抄骨架，按第 6 节清单推进；只有明确要整图厂商引擎时，才参考 pi05→TRT 的工作量预期

---

## 8. 关键源码索引

| 主题 | 路径 |
|------|------|
| Pattern 注册 | `tvm/python/tvm/relax/backend/pattern_registry.py` |
| 通用 pattern 片段 | `tvm/python/tvm/relax/backend/patterns.py` |
| cuBLAS 分区 | `tvm/python/tvm/relax/backend/cuda/cublas.py` |
| TensorRT 分区 | `tvm/python/tvm/relax/backend/contrib/tensorrt.py` |
| TensorRT converters | `tvm/src/runtime/extra/contrib/tensorrt/tensorrt_ops.cc` |
| TensorRT builder / runtime | `.../tensorrt_builder.cc`, `tensorrt_runtime.cc` |
| TensorRT Relax codegen | `tvm/src/relax/backend/contrib/tensorrt/codegen.cc` |
| RunCodegen pass | `tvm/python/tvm/relax/transform/transform.py` → `RunCodegen` |
| 教学 NPU | `tvm/python/tvm/relax/backend/contrib/example_npu/` |
| CMake 开关 | `tvm/cmake/config.cmake`（`USE_CUBLAS` / `USE_TENSORRT_*` / …） |
| pi05 落地文档 | `mlc-vla/docs/byoc/offload_trt.md` |
