"""The trainable alignment modules: projection + resampler.

These are the ONLY parameters that train. The ECG encoder and the LLM are both
frozen; gradients still flow through the frozen LLM to reach these modules.

Parameter counts for d_enc=768, d_llm=896, n_queries=32, n_heads=8:
    projection   768*896 + 896            =   689,024
    queries      32*896                   =    28,672
    attention    4 * (896*896) + biases   = ~3,214,000
    layernorm    2*896                    =     1,792
    ---------------------------------------------------
    total                                 ~ 3.9M trainable
"""

from typing import Optional

import torch
import torch.nn as nn


class Resampler(nn.Module):
    """Query-based cross-attention pooling (Perceiver / Flamingo / Q-Former family).

    A fixed set of learnable queries attends over a variable-length encoder
    output and emits a fixed number of vectors. Because cross-attention consumes
    a variable-length key/value set, sequence length is never a compatibility
    problem: 156 tokens or 251 tokens both produce n_queries outputs.

    What training teaches it: which directions in ECG representation space map
    to which directions in the LLM's embedding space.
    """

    def __init__(
        self,
        dim: int,
        n_queries: int = 32,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ln = nn.LayerNorm(dim)

    def forward(
        self,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """(batch, seq_len, dim) -> (batch, n_queries, dim)"""
        q = self.queries.unsqueeze(0).expand(kv.size(0), -1, -1)
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        return self.ln(out)


class Aligner(nn.Module):
    """projection + resampler, the full trainable bridge.

    encoder_out (batch, S, d_enc) -> soft prompt (batch, n_queries, d_llm)
    """

    def __init__(
        self,
        d_enc: int = 768,
        d_llm: int = 896,
        n_queries: int = 32,
        n_heads: int = 8,
    ):
        super().__init__()
        self.proj = nn.Linear(d_enc, d_llm)
        self.resampler = Resampler(d_llm, n_queries=n_queries, n_heads=n_heads)

    def forward(self, encoder_out: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.resampler(self.proj(encoder_out), key_padding_mask)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- open design questions to settle with the lab ------------------------
# 1. Projection before or after the resampler? Currently before, so attention
#    runs in d_llm. Running it in d_enc and projecting after is cheaper.
# 2. n_queries: 32 chosen arbitrarily. What does ChatHealthAI use?
# 3. Task-aware conditioning: ChatHealthAI's resampler is described as
#    "task-aware". Should the queries be conditioned on the question text?
# 4. Should the projection have a non-linearity / be an MLP rather than Linear?
