# 优化方向 01：Paged KV Cache + prefill/decode 拆分

本目录是对基线 [`../simple_llm_decoder.py`](../simple_llm_decoder.py) 的**第一步优化**。

两个脚本共用的模型骨架（`ModelConfig` / `GatedMLP` / `DecoderLayer` / RMSNorm）都抽到了 [`../common_decoder.py`](../common_decoder.py)，因此基线和本目录的脚本里**只保留各自差异的部分**，直接 diff 两个文件即可看清优化点。两者现在使用**同一个 `ModelConfig`**（同架构、同尺寸），差别纯粹在于执行策略。

后续每引入一个新的优化方向，就在 `tvm_tutorials/` 下新增一个同级目录（目录名即优化方向），逐步把一个"能跑 forward 的原型"演进成"可高效自回归生成的推理引擎"。参考蓝本：`$TVM_HOME/docs/how_to/tutorials/optimize_llm.py` 与 `mlc-llm/python/mlc_llm/model/llama/llama_model.py`。

---

## 这一步做了什么

| | 基线 `simple_llm_decoder.py` | 本目录 `decoder_paged_kv.py` |
|--|--|--|
| 推理接口 | 单一 `forward(input_ids)`，静态 `seq_len` | `embed` / `prefill` / `decode` 三段式 |
| Attention | 手写 `matmul → causal mask → softmax → matmul` | `PagedKVCache.attention_with_fused_qkv`（mask/softmax/RoPE 都在内核里） |
| KV 状态 | 无，每步都把整段序列重算 | 分页 KV cache，decode 只算新 token 对已缓存 K/V 的注意力 |
| 位置编码 | 无 | RoPE（`RopeMode.NORMAL`，写入 cache 前对 k 施加） |
| 生成方式 | 一次 forward + argmax | prefill 首 token + decode 循环逐 token |
| 依赖 | stock TVM | stock TVM（`TIRPagedKVCache` 直接实例化，**不需要 mlc_llm 的编译 pass**） |

核心思想和 MLC-LLM 一致：**Attention 层不再自己维护 K/V 张量**，而是把 `(layer_id, 融合的 qkv)` 交给一个有状态的 `PagedKVCache` 运行时对象；prefill 处理整段 prompt 并把 K/V 写进 cache，decode 每次只喂 1 个 token。

## 关键实现点

- `create_tir_paged_kv_cache`：直接构造 `TIRPagedKVCache`（`attn_kind="mha"`），导出后 IRModule 里会自动带上 `batch_prefill_paged_kv` / `batch_decode_paged_kv` / `fused_rope` / `tir_kv_cache_transpose_append` 等一整套 TIR 内核。
- `get_default_spec`：给 `embed / prefill / decode / create_tir_paged_kv_cache` 定义 Relax 函数签名，其中 `paged_kv_cache` 是 `nn.spec.Object`。
- `opt_llm` 编译流水线：在 `zero` pipeline 的 legalize+fuse 基础上，补了 `FuseTransposeMatmul`、DLight GPU schedule，以及 KV-cache 必需的 lowering pass（`CallTIRRewrite / StaticPlanBlockMemory / RewriteCUDAGraph / LowerRuntimeBuiltin / VMShapeLower …`）。
- 运行时驱动协议（每一步 forward 前后成对调用）：
  1. `vm.builtin.kv_state_add_sequence`（新建序列，仅一次）
  2. `vm.builtin.kv_state_begin_forward(cache, [seq_id], [append_len])`
  3. `vm["prefill" | "decode"](hidden, cache, params)`
  4. `vm.builtin.kv_state_end_forward`

## 运行

```bash
export TVM_HOME=/home/zhangxa/codes/edgeLLM/tvm
export TVM_LIBRARY_PATH=$TVM_HOME/build/lib
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
python3 decoder_paged_kv.py
```

预期输出：打印导出的 Relax 函数列表、prefill 的首 token，以及 decode 循环生成的 token 序列（随机权重，token 无语义，仅验证流程与内核正确执行）。

> 注：`common_decoder.py` 里统一用 `hidden_size=256 / head_dim=64`。因为 KV cache 的 TIR attention/RoPE 内核对 `head_dim` 有 tiling 约束，`head_dim=64` 是被充分测试过的安全取值；基线用手写 matmul，对尺寸不敏感，两者共用同一 config 不影响正确性。

## 后续优化方向（规划中的同级目录）

按收益 / 依赖顺序，未来可逐个新增目录：

- `fp16_weights/`：`model.to("float16")` + fp16 KV cache，降带宽（decode 通常 memory-bound）。
- `group_quant/`：int4 group 量化权重 + dequantize-matmul 融合（对齐 mlc-llm 的 `q4f16_1`）。
- `cuda_graph/`：坐实 `RewriteCUDAGraph`，测量 decode 单步 launch 开销下降。
- `flashinfer/`：CUDA 上把 `TIRPagedKVCache` 换成 `FlashInferPagedKVCache`。
- `continuous_batching/`：多序列 `add_sequence` + `batch_prefill`/`batch_decode`，走向 MLCEngine 式吞吐。
- `gqa/`：`num_kv_heads < num_heads`，共享 KV 头。
