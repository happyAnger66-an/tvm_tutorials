# TVM Tutorials：从朴素 Decoder 到 Paged KV 推理

用 Apache TVM / Relax 手写一个极小 LLM decoder，并在 CUDA 上逐步引入推理优化。目标不是训练或部署完整模型，而是把「能跑 forward」演进成「可高效自回归生成」的可运行脚手架，方便对照 MLC-LLM / TVM `optimize_llm` 一类路径。

## 项目做什么

1. 用 Relax `nn.Module` 定义共享 decoder 骨架（Embedding / Attention / GatedMLP / RMSNorm / LM Head）。
2. **基线**：无 KV cache，整段序列每次重算，验证 export → compile → CUDA 运行。
3. **优化 01**：Paged KV Cache + `prefill` / `decode` 拆分，decode 只算新 token 对已缓存 K/V 的注意力。
4. 每一步优化单独成目录，与基线 diff 即可看清差异；共用配置保证对比公平。

模型刻意做得很小（随机权重），输出 token 无语义，仅用于验证编译与运行时路径。

## 目录结构

```text
tvm_tutorials/
├── common_decoder.py          # 共享骨架：ModelConfig / RMSNorm / GatedMLP / DecoderLayer
├── simple_llm_decoder.py      # 基线：单一 forward，无 KV cache
├── learn_compare.py           # 学习工具：四块源码 diff + 分阶段 IR
├── paged_kv_cache/            # 优化 01：Paged KV + prefill/decode
│   ├── decoder_paged_kv.py
│   └── README.md
├── docs/
│   └── custom_tvm.md          # 端侧选型：TVM vs TensorRT / Inductor 等
├── install.md                 # 本机从源码编译 TVM（CUDA / cuDNN / CUTLASS / LLVM）
└── README.md
```

## 基线 vs Paged KV

| | `simple_llm_decoder.py` | `paged_kv_cache/decoder_paged_kv.py` |
|--|--|--|
| 推理接口 | 单一静态 `forward(input_ids)` | `embed` / `prefill` / `decode` |
| Attention | 手写 matmul → mask → softmax → matmul | `PagedKVCache.attention_with_fused_qkv` |
| KV 状态 | 无，每步整段重算 | 分页 KV cache |
| 位置编码 | 无 | RoPE（`RopeMode.NORMAL`） |
| 生成方式 | 一次 forward + 末位 argmax | prefill 首 token + decode 循环 |
| 编译流水线 | stock `zero` + DLight | 自定义 `opt_llm`（含 KV-cache lowering） |
| 依赖 | stock TVM | stock TVM（直接构造 `TIRPagedKVCache`，无需 mlc_llm） |

## 环境

需要已从源码编译好的 TVM（含 CUDA）。安装步骤见 [install.md](install.md)。

```bash
export TVM_HOME=/home/zhangxa/codes/edgeLLM/tvm
export TVM_LIBRARY_PATH=$TVM_HOME/build/lib
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```

本机 CUDA 13.2 下默认 NVRTC 可能遇到头文件问题，脚本已设置：

```bash
export TVM_CUDA_COMPILE_MODE=nvcc   # 脚本内也有 setdefault
```

## 运行

**基线：**

```bash
python3 simple_llm_decoder.py
```

预期：打印 Relax `forward` 签名、logits shape，以及 `CUDA decoder run OK`。

**Paged KV：**

```bash
python3 paged_kv_cache/decoder_paged_kv.py
```

预期：打印导出的 Relax 函数列表、prefill 首 token，以及 decode 生成的 token 序列，最后 `Paged KV cache prefill/decode run OK`。

## 学习工具：`learn_compare.py`

对照基线与 Paged KV，以及观察编译各阶段 IR：

```bash
# 1) 只盯四块源码差异：attention / export / pipeline / runtime
python3 learn_compare.py diff
python3 learn_compare.py diff --block attention
python3 learn_compare.py diff --side-by-side --out /tmp/tvm_diff

# 2) 分阶段对比 IR（默认：一览表 + 相邻差分 + TIR 样例，不刷全文）
python3 learn_compare.py stages --model baseline
python3 learn_compare.py stages --model paged --func prefill
python3 learn_compare.py stages --model baseline --show-tir matmul
python3 learn_compare.py stages --model both --out /tmp/tvm_ir   # SUMMARY + 各阶段全文落盘

# 需要完整 IR 时
python3 learn_compare.py stages --model baseline --dump-ir

# 两个功能一起跑
python3 learn_compare.py all --out /tmp/tvm_learn
```

`stages` 输出结构：① 各阶段数字表（tir/sched/fused/call_tir…）② 相邻阶段 Δ ③ Relax 关键行采样 ④ fuse→dlight 的 TIR 片段 diff。完整 IR 用 `--out` / `--dump-ir` 落盘再看。

## 设计约定

- 共享逻辑放在 `common_decoder.py`；各教程脚本只保留与优化相关的差异（attention、接口、编译流水线、运行时循环）。
- 基线与优化版共用同一 `ModelConfig`（同架构、同尺寸），差别在执行策略。
- `head_dim=64`：Paged KV 的 TIR attention / RoPE 内核对维度有 tiling 约束，取充分测试过的安全值。
- 后续每引入一个优化方向，在仓库根下新增同级目录（目录名即优化方向）。

## 规划中的后续目录

按收益 / 依赖顺序：

- `fp16_weights/`：fp16 权重与 KV cache，降 decode 带宽
- `group_quant/`：int4 group 量化 + dequantize-matmul 融合
- `cuda_graph/`：坐实 `RewriteCUDAGraph`，降 decode launch 开销
- `flashinfer/`：`TIRPagedKVCache` → `FlashInferPagedKVCache`
- `continuous_batching/`：多序列 batch prefill/decode
- `gqa/`：`num_kv_heads < num_heads`

## 参考

- `$TVM_HOME/docs/how_to/tutorials/optimize_llm.py`
- `mlc-llm/python/mlc_llm/model/llama/llama_model.py`
- [docs/custom_tvm.md](docs/custom_tvm.md)：开放 IR 编译框架与黑盒推理引擎的选型对比
