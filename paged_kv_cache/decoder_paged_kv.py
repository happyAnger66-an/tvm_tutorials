"""LLM decoder with a Paged KV Cache, split into prefill / decode (CUDA).

This is the first optimization step over ``../simple_llm_decoder.py``.

Shared model scaffolding (ModelConfig / GatedMLP / DecoderLayer / RMSNorm) lives
in ``../common_decoder.py`` and is identical to the baseline. This file only
spells out what the optimization changes, so a diff against the baseline is
exactly the delta:

  - Attention : ``PagedKVCache.attention_with_fused_qkv`` -- causal mask,
                softmax and RoPE all live inside the cache's attention kernel;
                past K/V are kept in a paged cache instead of being recomputed.
  - Interface : ``embed`` / ``prefill`` / ``decode`` + ``create_tir_paged_kv_cache``
                instead of a single static ``forward``.
  - Compile   : the "opt_llm" pipeline (adds KV-cache lowering + CUDA graph).
  - Runtime   : prefill the prompt, then decode one token at a time, driving the
                cache with the ``vm.builtin.kv_state_*`` protocol.

Attention is delegated to the KV cache: decode only computes attention for the
single new token against the cached K/V. ``create_tir_paged_kv_cache``
instantiates ``TIRPagedKVCache`` directly, so this script needs *only* stock
TVM -- no mlc_llm compile passes.

Reference: ``$TVM_HOME/docs/how_to/tutorials/optimize_llm.py`` and
``mlc-llm/python/mlc_llm/model/llama/llama_model.py``.

Usage:
    export TVM_HOME=/home/zhangxa/codes/edgeLLM/tvm
    export TVM_LIBRARY_PATH=$TVM_HOME/build/lib
    export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
    python3 decoder_paged_kv.py
"""

import os

# CUDA 13.2 + default NVRTC has header issues (cuda_fp8.hpp); use nvcc backend.
os.environ.setdefault("TVM_CUDA_COMPILE_MODE", "nvcc")

import enum
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from tvm_ffi import Shape

import tvm
from tvm import relax, te, tirx
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache, TIRPagedKVCache
from tvm.s_tir import dlight as dl

from common_decoder import DecoderLayer, ModelConfig, rms_norm


class RopeMode(enum.IntEnum):
    """RoPE application mode of the Paged KV cache.

    NONE   : do not apply RoPE.
    NORMAL : apply RoPE to k before writing k into the cache.
    INLINE : apply RoPE to q/k inside the attention kernel on the fly.
    """

    NONE = 0
    NORMAL = 1
    INLINE = 2


class CausalSelfAttention(nn.Module):
    """Fused-QKV attention that reads/writes the paged KV cache.

    Unlike the baseline, there is no explicit causal mask or softmax here: both
    live inside the KV cache attention kernel, together with RoPE.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x: Tensor, paged_kv_cache: PagedKVCache, layer_id: int) -> Tensor:
        d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
        b, s, _ = x.shape
        qkv = self.qkv_proj(x)  # [b, s, (h_q + 2*h_kv) * d]
        qkv = op.reshape(qkv, (b, s, h_q + 2 * h_kv, d))
        attn = paged_kv_cache.attention_with_fused_qkv(
            layer_id, qkv, self.num_q_heads, sm_scale=self.head_dim**-0.5
        )
        out = op.reshape(attn, (b, s, h_q * d))
        return self.o_proj(out)


class DecoderModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, CausalSelfAttention(config)) for _ in range(config.num_layers)]
        )
        self.final_norm = rms_norm(config)

    def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache) -> Tensor:
        hidden = input_embed
        for layer_id, layer in enumerate(self.layers):
            hidden = layer(hidden, paged_kv_cache, layer_id)
        return self.final_norm(hidden)


class SimpleDecoder(nn.Module):
    """Same model family as the baseline, but organized as a KV-cache-driven
    prefill/decode engine.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model = DecoderModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.rope_theta = config.rope_theta
        self.dtype = config.dtype

    def to(self, dtype=None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids: Tensor) -> Tensor:
        return self.model.embed(input_ids)

    def get_logits(self, hidden_states: Tensor) -> Tensor:
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        # Run the full prompt, then keep only the last token's hidden state.
        def _index(x: te.Tensor):  # x[:, -1:, :]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden = self.model(input_embed, paged_kv_cache)
        hidden = op.tensor_expr_op(_index, name_hint="index", args=[hidden])
        logits = self.get_logits(hidden)
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        hidden = self.model(input_embed, paged_kv_cache)
        logits = self.get_logits(hidden)
        return logits, paged_kv_cache

    def create_tir_paged_kv_cache(
        self,
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
    ) -> PagedKVCache:
        return TIRPagedKVCache(
            attn_kind="mha",
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=0,
            layer_partition=relax.ShapeExpr([0, self.num_layers]),
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            mla_original_qk_head_dim=0,
            mla_original_v_head_dim=0,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.rope_theta,
            rope_scaling={},
            rope_ext_factors=relax.PrimValue(0),
            rotary_dim=self.head_dim,
            dtype=self.dtype,
            target=target,
            enable_disaggregation=False,
        )

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {"param_mode": "packed", "effect_mode": "none"},
            },
            "create_tir_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "$": {"param_mode": "none", "effect_mode": "none"},
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)


@register_pipeline("opt_llm")
def _pipeline(ext_mods=None):
    ext_mods = ext_mods or []

    @tvm.transform.module_pass(opt_level=0)
    def _seq(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
                # Phase 1: high-level graph fusion
                relax.transform.FuseTransposeMatmul(),
                # Phase 2: lower to TIR (the stock "zero" pipeline body)
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FoldConstant(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
                # Phase 3: TIR-level cleanups
                relax.transform.DeadCodeElimination(),
                # Phase 4: DLight GPU schedules
                dl.ApplyDefaultSchedule(
                    dl.gpu.Matmul(),
                    dl.gpu.GEMV(),
                    dl.gpu.Reduction(),
                    dl.gpu.GeneralReduction(),
                    dl.gpu.Fallback(),
                ),
                # Phase 5: lower to VM bytecode (incl. KV-cache builtins & CUDA graph)
                relax.transform.RewriteDataflowReshape(),
                relax.transform.ToNonDataflow(),
                relax.transform.RemovePurityChecking(),
                relax.transform.CallTIRRewrite(),
                relax.transform.StaticPlanBlockMemory(),
                relax.transform.RewriteCUDAGraph(),
                relax.transform.LowerAllocTensor(),
                relax.transform.KillAfterLastUse(),
                relax.transform.LowerRuntimeBuiltin(),
                relax.transform.VMShapeLower(),
                relax.transform.AttachGlobalSymbol(),
                relax.transform.AttachExternModules(ext_mods),
            ]
        )
        return seq(mod)

    return _seq


dev = tvm.device("cuda", 0)
target = tvm.target.Target.from_device(dev)


def main():
    config = ModelConfig()
    model = SimpleDecoder(config)

    # 1. Export the nn.Module (embed / prefill / decode / create_tir_paged_kv_cache).
    mod, named_params = model.export_tvm(spec=model.get_default_spec())
    print("=== Exported Relax functions ===")
    print(sorted(gv.name_hint for gv in mod.functions))

    # 2. Compile with the LLM pipeline.
    with target:
        ex = tvm.compile(mod, target, relax_pipeline=relax.get_pipeline("opt_llm"))
    vm = relax.VirtualMachine(ex, dev)

    # 3. Random weights on the GPU.
    rng = np.random.default_rng(0)
    params = [
        tvm.runtime.tensor(rng.standard_normal(p.shape).astype(p.dtype) * 0.02, dev)
        for _, p in named_params
    ]

    # 4. KV cache + runtime helpers.
    add_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    begin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    end_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    reshape = tvm.get_global_func("vm.builtin.reshape")

    kv_cache = vm["create_tir_paged_kv_cache"](
        Shape([1]),  # max_batch_size
        Shape([config.context_window_size]),  # max_total_seq_len
        Shape([config.prefill_chunk_size]),  # prefill_chunk_size
        Shape([16]),  # page_size
    )

    def embed(token_ids: np.ndarray) -> tvm.runtime.Tensor:
        toks = tvm.runtime.tensor(token_ids.astype("int32"), dev)
        h = vm["embed"](toks, params)  # [seq_len, hidden]
        return reshape(h, Shape([1, h.shape[0], h.shape[1]]))  # [1, seq_len, hidden]

    def argmax_token(logits) -> int:
        return int(logits.numpy()[0, -1].argmax())

    # 5. Prefill on a random prompt.
    seq_id = 0
    prompt = rng.integers(0, config.vocab_size, size=(8,))
    add_sequence(kv_cache, seq_id)

    hidden = embed(prompt)
    begin_forward(kv_cache, Shape([seq_id]), Shape([int(prompt.shape[0])]))
    logits, kv_cache = vm["prefill"](hidden, kv_cache, params)
    end_forward(kv_cache)

    next_token = argmax_token(logits)
    generated = [next_token]
    print("\n=== Prefill ===")
    print("prompt tokens :", prompt.tolist())
    print("logits shape  :", logits.shape)  # (1, 1, vocab_size)
    print("first token   :", next_token)

    # 6. Decode loop: one token at a time, attending only over the cached K/V.
    num_decode_steps = 16
    for _ in range(num_decode_steps):
        hidden = embed(np.array([next_token]))
        begin_forward(kv_cache, Shape([seq_id]), Shape([1]))
        logits, kv_cache = vm["decode"](hidden, kv_cache, params)
        end_forward(kv_cache)
        next_token = argmax_token(logits)
        generated.append(next_token)

    print("\n=== Decode ===")
    print("generated tokens:", generated)
    print("Paged KV cache prefill/decode run OK")


if __name__ == "__main__":
    main()
