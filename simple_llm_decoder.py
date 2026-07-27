"""Minimal LLM decoder (BASELINE), compiled and run on CUDA.

Shared model scaffolding (ModelConfig / GatedMLP / DecoderLayer / RMSNorm) lives
in ``common_decoder.py``. This file only spells out the BASELINE choices, so a
diff against ``paged_kv_cache/decoder_paged_kv.py`` shows exactly what the
optimization changes:

  - Attention : hand-written matmul -> causal mask -> softmax -> matmul,
                no KV cache and no positional encoding.
  - Interface : a single static ``forward(input_ids)`` that recomputes the whole
                sequence on every call.
  - Compile   : stock "zero" pipeline + DLight GPU schedules.
  - Runtime   : one forward call, argmax the last position.

Usage:
    export TVM_HOME=/home/zhangxa/codes/edgeLLM/tvm
    export TVM_LIBRARY_PATH=$TVM_HOME/build/lib
    export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
    python3 simple_llm_decoder.py
"""

import os

# TVM 生成 CUDA kernel 时默认走 NVRTC；CUDA 13.2 下 NVRTC 可能缺 cuda_fp8.hpp。
# 设为 "nvcc" 则改用系统 nvcc 编译，避开该头文件问题。setdefault 表示已有环境变量时不覆盖。
os.environ.setdefault("TVM_CUDA_COMPILE_MODE", "nvcc")

import sys

# 把本文件所在目录加入 sys.path，保证能 import 同级的 common_decoder.py。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import tvm  # TVM 顶层：device / Target / compile / runtime.tensor 等
from tvm import relax  # Relax：图级 IR + 编译流水线 + VirtualMachine 运行时
from tvm.relax.frontend import nn  # Relax 前端的 nn.Module API（写法接近 PyTorch）
from tvm.relax.frontend.nn import Tensor, op  # Tensor：符号张量；op：算子（matmul/reshape/...）
from tvm.s_tir import dlight as dl  # DLight：为 TIR 算子自动套用 GPU 默认 schedule

from common_decoder import DecoderLayer, ModelConfig, rms_norm


class CausalSelfAttention(nn.Module):
    """手写 causal self-attention（基线）。

    每次 forward 都对整段序列重算：matmul → causal mask → softmax → matmul。
    无 KV cache、无 RoPE；对比优化版可看清差异。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads  # 注意力头数
        self.head_dim = config.head_dim  # 每个头的维度；hidden = num_heads * head_dim
        self.scale = self.head_dim**-0.5  # 缩放因子 1/sqrt(d)，稳定 softmax
        # nn.Linear 在 export 时会变成 Relax matmul + 可训练参数（权重）。
        # Q/K/V 融合成一次投影：输出通道为 3 * hidden_size。
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        # 多头注意力拼回后的输出投影。
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: Relax 符号张量，形状 [batch, seq_len, hidden]。此处尚未执行，只是建计算图。
        b, s, h = x.shape  # 解包动态/静态形状（export 时 batch=1, s=seq_len）

        # 线性投影得到融合的 QKV，形状 [b, s, 3*h]。
        qkv = self.qkv_proj(x)
        # 拆成 [b, s, 3, num_heads, head_dim]，便于沿第 2 维切成 Q/K/V。
        qkv = op.reshape(qkv, (b, s, 3, self.num_heads, self.head_dim))
        # op.split：沿 axis=2 切成 3 份；每份 [b, s, 1, num_heads, head_dim]。
        q, k, v = op.split(qkv, 3, axis=2)

        # 去掉 split 留下的 size=1 轴，再把 heads 维挪到前面，得到 [b, num_heads, s, head_dim]。
        # permute_dims 的索引列表是“新位置上来自旧位置的哪一维”。
        q = op.permute_dims(op.reshape(q, (b, s, self.num_heads, self.head_dim)), [0, 2, 1, 3])
        k = op.permute_dims(op.reshape(k, (b, s, self.num_heads, self.head_dim)), [0, 2, 1, 3])
        v = op.permute_dims(op.reshape(v, (b, s, self.num_heads, self.head_dim)), [0, 2, 1, 3])

        # scores = Q @ K^T，再乘 scale；结果 [b, heads, s, s]。
        # permute_dims(k, [0,1,3,2]) 把最后两维转置，得到 K^T。
        scores = op.matmul(q, op.permute_dims(k, [0, 1, 3, 2])) * self.scale

        # 上三角因果 mask：j > i 的位置填 -1e4，softmax 后近似为 0，禁止看未来 token。
        # op.full 造常数矩阵；op.triu(..., diagonal=1) 取严格上三角。
        mask = op.triu(op.full((s, s), -1.0e4, dtype=x.dtype), diagonal=1)
        scores = scores + mask  # 广播加到每个 batch/head 的 attention score 上

        # 沿最后一维（key 位置）做 softmax，得到注意力权重。
        attn = op.softmax(scores, axis=-1)
        # 加权求和：attn @ V → [b, heads, s, head_dim]
        out = op.matmul(attn, v)
        # 把头维拼回 hidden：先 permute 成 [b, s, heads, head_dim]，再 reshape 成 [b, s, h]
        out = op.reshape(op.permute_dims(out, [0, 2, 1, 3]), (b, s, h))
        # 输出投影，回到 [b, s, hidden]
        return self.o_proj(out)


class SimpleDecoder(nn.Module):
    """完整 decoder：Embedding → N × DecoderLayer → RMSNorm → LM Head。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        # token id → hidden 向量的查表层（export 后是 Gather / take 类算子）。
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        # ModuleList：可导出的层列表；每层注入本文件的 CausalSelfAttention。
        self.layers = nn.ModuleList(
            [DecoderLayer(config, CausalSelfAttention(config)) for _ in range(config.num_layers)]
        )
        self.final_norm = rms_norm(config)  # 最后一层 RMSNorm
        # 隐状态 → vocab logits；bias=False 与常见 LLM 一致。
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids: [batch, seq_len] int32 token 序列
        x = self.embed(input_ids)  # → [batch, seq_len, hidden]
        for layer in self.layers:
            x = layer(x)  # 每层：pre-norm attn + residual，再 pre-norm MLP + residual
        x = self.final_norm(x)
        return self.lm_head(x)  # → [batch, seq_len, vocab_size]


def main():
    config = ModelConfig()  # 默认小模型超参（见 common_decoder.py）
    model = SimpleDecoder(config)  # 此时还是 Python 前端图，尚未编译

    # ------------------------------------------------------------------
    # 1) 导出：把 nn.Module 转成 Relax IRModule + 参数名/形状列表
    # ------------------------------------------------------------------
    # export_tvm 会根据 spec 做符号追踪（类似 torch.fx / jax.make_jaxpr）：
    #   - 键 "forward" 对应要导出的方法名
    #   - nn.spec.Tensor([...], dtype) 声明该入参的静态形状与类型
    # 这里固定 [1, seq_len]，基线不做动态长度。
    # 返回：
    #   mod          : tvm.IRModule，内含 Relax 函数（至少有 "forward"）
    #   named_params : [(name, placeholder), ...]，按调用时参数顺序排列
    mod, named_params = model.export_tvm(
        spec={
            "forward": {
                "input_ids": nn.spec.Tensor([1, config.seq_len], "int32"),
            }
        }
    )
    print("=== Exported Relax IRModule (forward signature) ===")
    # Function.show()：打印该 Relax 函数的文本 IR（便于检查图结构）。
    mod["forward"].show()

    # ------------------------------------------------------------------
    # 2) 编译：选 CUDA 设备与 Target，跑优化 pass，再 codegen
    # ------------------------------------------------------------------
    # tvm.device("cuda", 0)：第 0 号 GPU 运行时设备（决定张量放哪、VM 在哪跑）。
    dev = tvm.device("cuda", 0)
    # Target 描述“为谁生成代码”（架构、特性）；from_device 从当前设备探测。
    target = tvm.target.Target.from_device(dev)

    # with target: 把当前 Target 设为上下文，部分 pass / schedule 会读取它。
    with target:
        # Sequential：把多个 IRModule pass 串成一条流水线，依次作用在 mod 上。
        mod = tvm.ir.transform.Sequential(
            [
                # "zero" pipeline：stock 最小图优化管线，典型步骤包括
                # LegalizeOps（高级算子 → TIR/可执行形式）、AnnotateTIROpPattern、
                # FoldConstant、FuseOps、FuseTIR 等（legalize + 算子融合）。
                relax.get_pipeline("zero"),
                # DLight：按算子类别套默认 GPU schedule（提高并行度/访存）。
                # 列表顺序即匹配优先级；Fallback 兜底处理未特化的算子。
                dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),  # 稠密矩阵乘
                    dl.gpu.GEMV(),  # 矩阵-向量乘（decode 常见）
                    dl.gpu.Reduction(),  # 归约类
                    dl.gpu.GeneralReduction(),  # 更通用的归约模式
                    dl.gpu.Fallback(),  # 其余算子的通用 GPU schedule
                ),
            ]
        )(mod)  # Sequential(...) 返回可调用对象；(mod) 真正执行变换

    # tvm.compile：把优化后的 IRModule 编译成可执行模块（含 CUDA kernel）。
    # 返回的 Executable 可交给 Relax VirtualMachine 加载。
    ex = tvm.compile(mod, target=target)
    # VirtualMachine：解释/执行 Relax VM 字节码；绑定到 dev，GPU kernel 在该设备上跑。
    vm = relax.VirtualMachine(ex, dev)

    # ------------------------------------------------------------------
    # 3) 准备 GPU 上的随机权重与输入（本教程不加载真实 checkpoint）
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)  # 固定种子，结果可复现
    # named_params 里每个 p 有 .shape / .dtype；转成 numpy 后拷到 CUDA。
    # tvm.runtime.tensor(ndarray, device)：创建设备上的 NDArray（运行时张量）。
    # * 0.02：小随机初始化，避免 logits 过大。
    gpu_params = [
        tvm.runtime.tensor(rng.standard_normal(p.shape).astype(p.dtype) * 0.02, dev)
        for _, p in named_params
    ]
    # 随机 token id，形状与 export spec 一致：[1, seq_len]，dtype int32。
    input_ids = tvm.runtime.tensor(
        rng.integers(0, config.vocab_size, size=(1, config.seq_len)).astype("int32"), dev
    )

    # ------------------------------------------------------------------
    # 4) 调用编译好的 forward，检查 logits
    # ------------------------------------------------------------------
    # vm["forward"]：按全局符号名取到导出的函数。
    # 调用约定：先传用户输入，再按 named_params 顺序展开权重（*gpu_params）。
    # 返回仍是设备上的 runtime.Tensor；.numpy() 拷回 Host 便于打印。
    logits = vm["forward"](input_ids, *gpu_params).numpy()
    print("\n=== Forward run on CUDA ===")
    print("logits shape:", logits.shape)  # 期望 (1, seq_len, vocab_size)
    # 取最后一个位置的 argmax，模拟“预测下一个 token”（随机权重无语义）。
    next_token = int(logits[0, -1].argmax())
    print("predicted next token id:", next_token)
    print("CUDA decoder run OK")


if __name__ == "__main__":
    main()
