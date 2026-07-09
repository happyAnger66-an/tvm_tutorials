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

# CUDA 13.2 + default NVRTC has header issues (cuda_fp8.hpp); use nvcc backend.
os.environ.setdefault("TVM_CUDA_COMPILE_MODE", "nvcc")

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import tvm
from tvm import relax
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.s_tir import dlight as dl

from common_decoder import DecoderLayer, ModelConfig, rms_norm


class CausalSelfAttention(nn.Module):
    """Hand-written attention: every call recomputes scores over the full
    sequence (matmul -> causal mask -> softmax -> matmul). No KV cache, no RoPE.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, s, h = x.shape
        qkv = self.qkv_proj(x)  # [b, s, 3*h]
        qkv = op.reshape(qkv, (b, s, 3, self.num_heads, self.head_dim))
        q, k, v = op.split(qkv, 3, axis=2)  # each [b, s, 1, num_heads, head_dim]
        # drop the split axis and move heads forward -> [b, num_heads, s, head_dim]
        q = op.permute_dims(op.reshape(q, (b, s, self.num_heads, self.head_dim)), [0, 2, 1, 3])
        k = op.permute_dims(op.reshape(k, (b, s, self.num_heads, self.head_dim)), [0, 2, 1, 3])
        v = op.permute_dims(op.reshape(v, (b, s, self.num_heads, self.head_dim)), [0, 2, 1, 3])

        # scores[b, heads, s, s] = q @ k^T
        scores = op.matmul(q, op.permute_dims(k, [0, 1, 3, 2])) * self.scale

        # causal mask: future positions (j > i) get a large negative bias
        mask = op.triu(op.full((s, s), -1.0e4, dtype=x.dtype), diagonal=1)
        scores = scores + mask

        attn = op.softmax(scores, axis=-1)
        out = op.matmul(attn, v)  # [b, heads, s, head_dim]
        out = op.reshape(op.permute_dims(out, [0, 2, 1, 3]), (b, s, h))
        return self.o_proj(out)


class SimpleDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, CausalSelfAttention(config)) for _ in range(config.num_layers)]
        )
        self.final_norm = rms_norm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def main():
    config = ModelConfig()
    model = SimpleDecoder(config)

    # 1. Export the nn.Module to a Relax IRModule (single static forward).
    mod, named_params = model.export_tvm(
        spec={
            "forward": {
                "input_ids": nn.spec.Tensor([1, config.seq_len], "int32"),
            }
        }
    )
    print("=== Exported Relax IRModule (forward signature) ===")
    mod["forward"].show()

    # 2. Compile for CUDA: zero pipeline (legalize + fuse) then DLight GPU schedules.
    dev = tvm.device("cuda", 0)
    target = tvm.target.Target.from_device(dev)
    with target:
        mod = tvm.ir.transform.Sequential(
            [
                relax.get_pipeline("zero"),
                dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                ),
            ]
        )(mod)

    ex = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(ex, dev)

    # 3. Random weights + input token ids, all placed on the GPU.
    rng = np.random.default_rng(0)
    gpu_params = [
        tvm.runtime.tensor(rng.standard_normal(p.shape).astype(p.dtype) * 0.02, dev)
        for _, p in named_params
    ]
    input_ids = tvm.runtime.tensor(
        rng.integers(0, config.vocab_size, size=(1, config.seq_len)).astype("int32"), dev
    )

    # 4. Run forward and inspect the logits.
    logits = vm["forward"](input_ids, *gpu_params).numpy()
    print("\n=== Forward run on CUDA ===")
    print("logits shape:", logits.shape)  # (1, seq_len, vocab_size)
    next_token = int(logits[0, -1].argmax())
    print("predicted next token id:", next_token)
    print("CUDA decoder run OK")


if __name__ == "__main__":
    main()
