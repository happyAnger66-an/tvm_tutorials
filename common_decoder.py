"""Shared building blocks for the decoder tutorials.

Both the baseline (``simple_llm_decoder.py``) and the optimized versions (e.g.
``paged_kv_cache/decoder_paged_kv.py``) reuse the pieces defined here, so each
tutorial script only needs to spell out the part that actually *differs* -- the
attention implementation, the inference interface (``forward`` vs
``prefill``/``decode``), the compile pipeline and the runtime loop.

Everything in this file is identical across the tutorials:

  - ``ModelConfig``  : model hyper-parameters
  - ``rms_norm``     : RMSNorm factory
  - ``GatedMLP``     : SiLU-gated feed-forward network
  - ``DecoderLayer`` : pre-norm residual block (the attention module is injected,
                       and any extra attention arguments -- kv cache, layer id --
                       are forwarded transparently via ``*attn_args``)
"""

import dataclasses

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op


@dataclasses.dataclass
class ModelConfig:
    # --- shared architecture (same model across all tutorials) ---
    vocab_size: int = 256
    hidden_size: int = 256
    num_heads: int = 4
    num_kv_heads: int = 4  # MHA: == num_heads (set smaller for GQA)
    head_dim: int = 64  # hidden_size == num_heads * head_dim
    num_layers: int = 2
    intermediate_size: int = 512
    rms_norm_eps: float = 1e-5
    dtype: str = "float32"
    # --- baseline only: static sequence length for the single forward ---
    seq_len: int = 16
    # --- paged-kv only: RoPE base + KV cache capacity ---
    rope_theta: int = 10000
    context_window_size: int = 512
    prefill_chunk_size: int = 512


def rms_norm(config: ModelConfig) -> nn.RMSNorm:
    return nn.RMSNorm(config.hidden_size, axes=-1, epsilon=config.rms_norm_eps, bias=False)


class GatedMLP(nn.Module):
    """SiLU-gated FFN: down(silu(gate) * up), with gate/up fused into one matmul."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = op.split(gate_up, 2, axis=-1)
        return self.down_proj(op.silu(gate) * up)


class DecoderLayer(nn.Module):
    """Pre-norm residual block:

        x = x + attn(input_norm(x), *attn_args)
        x = x + mlp(post_norm(x))

    The attention module is injected at construction so each tutorial can plug in
    either the hand-written attention or the paged-KV attention. Extra arguments
    an attention needs (paged kv cache, layer id) flow through ``*attn_args``,
    which is empty for the baseline attention.
    """

    def __init__(self, config: ModelConfig, attention: nn.Module):
        super().__init__()
        self.input_norm = rms_norm(config)
        self.attn = attention
        self.post_norm = rms_norm(config)
        self.mlp = GatedMLP(config)

    def forward(self, x: Tensor, *attn_args) -> Tensor:
        x = x + self.attn(self.input_norm(x), *attn_args)
        x = x + self.mlp(self.post_norm(x))
        return x
