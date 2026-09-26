"""Task-aware multi-modal Perceiver resampler (ChatNT-style).

Faithful port of the lab's reference implementation (`reference_resampler.py`,
the ChatNT `TorchMultiModalPerceiverResampler` stack). Submodule attribute names
are kept identical (`bio_projection`, `perceiver_resampler.layers.N.
cross_attention_{1,2}.w_{q,k,v}`, `fc1`/`fc2`, `latent_queries`, ...) so a lab
checkpoint's `state_dict` loads without renaming.

Structure, per the reference:

    ecg_embeddings  (B, 312, 768)   ecg (bio) stream
        | input_norm: LayerNorm(768)                    [ON by default, see below]
        | bio_projection: 768 -> embed_dim              [only input projection]
        v
    projected_ecg   (B, 312, embed_dim)
    task_embeddings (B, T, embed_dim)   task (english) stream, from the frozen
                                        student LLM's embedding layer, already
                                        at embed_dim -- no projection.

    latent_queries  (64, embed_dim), init randn / sqrt(embed_dim)

    Each of the `num_layers` blocks, with x = the latents:
        stream 1 (ecg):  x = x + cross_attn_1(pre_ln(x), KV=[ecg ; x])
        stream 2 (task): x = x + cross_attn_2(pre_ln(x), KV=[task; x])
        ffn:             x = x + fc2(gelu(fc1(ln(x))))
      Each cross-attention appends the current latents to its own stream's KV
      (`torch.cat([stream, latents], dim=1)`); the padding mask appends
      `resampled_length` ones to match. There is NO output projection: the
      module already runs at embed_dim, so the (B, 64, embed_dim) latents are
      the output, ready to splice into the student's `inputs_embeds`.

Two normalisations are added on top of the reference, both ON by default and
both behind config flags, because the two ends of this stack meet tensors the
reference never had to handle.

`input_layer_norm` normalises the cached ECG-FM embeddings before
`bio_projection`. Measured over the cache: per-position L2 norms are 350.5 +- 1.2
against a student embedding scale of ~1.3. Without a norm, bio_projection has to
learn a ~262x contraction on top of the mapping itself, from an initialisation
scaled for neither. LayerNorm rather than a fixed divisor: the norms are so nearly
constant (0.04% spread within a record) that the two are numerically almost the
same thing here, but LayerNorm's bias is learnable, and it is the only one of the
two that CAN address what the same measurement turned up next --

    any two positions, across any two records, have cosine similarity ~0.97: about
    98.6% of each vector's magnitude is a single direction shared by every position
    of every record. Subtract the global mean and cosine similarity falls to ~0.01,
    with a residual norm of ~55 out of ~350.

LayerNorm does not remove that direction on its own (it centres each position
across components, and this is a fixed vector, not a per-component offset), but
its bias has the capacity to learn to cancel it; a scalar divisor does not.
Initialising that bias from the dataset mean would hand the model the subtraction
for free instead of making it learn one, which is worth trying and is NOT done
here.

    Finally, `final_output_norm` (ON by default, and the other departure from the
    reference) applies a LayerNorm to those latents. The reference returns the
    raw block output, whose scale is whatever the stack happens to produce; but
    these latents are spliced in front of the student's token embeddings, and
    attention logits are dot products, so a prefix much shorter than those
    embeddings collects almost no attention weight and the frozen LLM reads past
    the ECG whatever the latents encode. Calibrate the LayerNorm's gain to the
    student's own embedding scale before training:

        target = mean_embedding_norm(student.get_input_embeddings().weight)
        resampler.calibrate_output_norm(target)      # or calibrate_from_embeddings

    The gain is an ordinary trainable parameter afterwards -- this sets where
    training starts, it does not pin the scale. Set `final_output_norm=False` for
    the reference behaviour and to load any checkpoint trained without it.

Apart from those there are deliberately no `use_ffn` / `condition_on_task`
switches -- the reference has neither. The FFN is always present; conditioning is
always two streams. To ablate task conditioning, zero the task stream at the call
site; do not change the architecture.

Trainable parameter count (`param_breakdown`), ECG config
(embed_dim=5120, ffn_embed_dim=20480, key_size=320, attention_heads=16,
num_layers=2, resampled_length=64):

    input_norm (LayerNorm on the ECG)     1,536
    bio_projection                    3,937,280
    latent_queries                      327,680
    per block x2 :
        cross_attention_1           104,878,080
        cross_attention_2           104,878,080
        norms (3 x LayerNorm)            30,720
        fc1                         104,878,080
        fc2                         104,862,720
    output_norm (final LayerNorm)        10,240
    ---------------------------------------------
    total                           843,332,096

    (843,320,320 with both `input_layer_norm=False` and `final_output_norm=False`,
    the reference count.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


@dataclass
class PerceiverResamplerConfig:
    """Hyperparameters for the multi-modal Perceiver resampler.

    Fields mirror the reference config. `input_embed_dim` is the width of the
    raw ECG (bio) stream before `bio_projection` lifts it to `embed_dim`; every
    other tensor already lives at `embed_dim`.
    """

    # architecture
    attention_heads: int = 16
    key_size: Optional[int] = None
    embed_dim: int = 5120
    ffn_embed_dim: int = 20480
    num_layers: int = 2
    resampled_length: int = 64

    add_bias_kv: bool = False
    add_bias_ffn: bool = True
    ffn_activation_name: str = "gelu-no-approx"
    use_glu_in_ffn: bool = False

    # the ECG (bio) stream arrives at this width and is projected to embed_dim
    input_embed_dim: int = 768

    # LayerNorm on the cached ECG embeddings, before bio_projection. NOT in the
    # reference, which assumes an input already at a sane scale. ECG-FM's
    # encoder_out is not: measured over the cache, per-position norms are ~350
    # against a student embedding scale of ~1.3, so an unnormalised bio_projection
    # is asked to contract by ~262x at the same time as it learns the mapping, from
    # an initialisation that assumes neither. Set False for the reference behaviour
    # or to load a checkpoint trained without it.
    input_layer_norm: bool = True

    # Final LayerNorm on the latents, after the last block. NOT in the reference:
    # the reference returns the raw block output, whose scale is whatever the stack
    # happens to produce. Spliced in front of the student's token embeddings, a
    # prefix much shorter than those embeddings draws almost no attention weight
    # (attention logits are dot products), so the frozen LLM reads past the ECG no
    # matter what the latents encode. This norm puts the prefix on the student's
    # own scale. Set False to reproduce the reference / any pre-norm checkpoint.
    final_output_norm: bool = True
    # Mean per-position L2 norm to target, normally the student's input-embedding
    # scale from :func:`mean_embedding_norm`. None leaves the LayerNorm at its
    # default unit weight, i.e. an output norm of about sqrt(embed_dim).
    output_target_norm: Optional[float] = None

    def __post_init__(self) -> None:
        if self.key_size is None:
            if self.embed_dim % self.attention_heads != 0:
                raise ValueError(
                    "When no key size is provided, embed_dim must be divisible "
                    f"by attention_heads, got embed_dim={self.embed_dim} and "
                    f"attention_heads={self.attention_heads}."
                )
            self.key_size = self.embed_dim // self.attention_heads

    @classmethod
    def ecg(cls, **overrides) -> "PerceiverResamplerConfig":
        """The training config: DeepSeek-R1-Distill-Qwen-14B student, d=5120."""
        return cls(
            embed_dim=5120,
            ffn_embed_dim=20480,
            key_size=320,
            attention_heads=16,
            num_layers=2,
            resampled_length=64,
            input_embed_dim=768,
            **overrides,
        )

    @classmethod
    def debug(cls, **overrides) -> "PerceiverResamplerConfig":
        """A CPU-sized stand-in: Qwen2.5-0.5B student width, d=896."""
        return cls(
            embed_dim=896,
            ffn_embed_dim=3584,
            key_size=56,
            attention_heads=16,
            num_layers=2,
            resampled_length=64,
            input_embed_dim=768,
            **overrides,
        )


def mean_embedding_norm(embedding_weight: torch.Tensor) -> float:
    """Mean per-row L2 norm of an embedding matrix, computed once.

    Pass the frozen student's input-embedding weight ``(vocab, embed_dim)``. The
    answer is the scale the spliced latents have to live at to compete with real
    tokens for attention. Computed in fp32 under ``no_grad``: the student is frozen
    and the matrix is bf16, where the sum of 5120 squares loses precision.
    """
    with torch.no_grad():
        return float(embedding_weight.detach().float().norm(dim=-1).mean())


class MultiHeadAttention(nn.Module):
    """Reference multi-head attention with a custom (per-head) key size.

    A boolean `attention_mask` is True where a key may be attended to; masked
    positions get -1e30 before the softmax.
    """

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        add_bias_kv: bool = False,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
    ):
        super().__init__()
        if not model_size:
            model_size = key_size * num_heads
        if not value_size:
            value_size = key_size
        self.model_size = model_size
        self.key_size = key_size
        self.value_size = value_size
        self.add_bias_kv = add_bias_kv
        self.num_heads = num_heads

        self.w_k = nn.Linear(self.model_size, self.num_heads * self.key_size)
        self.w_q = nn.Linear(self.model_size, self.num_heads * self.key_size)
        self.w_v = nn.Linear(self.model_size, self.num_heads * self.value_size)
        self.output = nn.Linear(self.num_heads * self.value_size, self.model_size)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_weight_bias: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        key_heads = self.w_k(key).reshape(
            (*key.shape[:-1], self.num_heads, self.key_size)
        )
        query_heads = self.w_q(query).reshape(
            (*query.shape[:-1], self.num_heads, self.key_size)
        )
        value_heads = self.w_v(value).reshape(
            (*value.shape[:-1], self.num_heads, self.value_size)
        )
        attention_weights = torch.einsum(
            "...thd, ...Thd -> ...htT", query_heads, key_heads
        )
        attention_weights = attention_weights / np.sqrt(self.key_size)
        if attention_mask is not None:
            attention_weights = torch.where(attention_mask, attention_weights, -1e30)

        attention_weights = attention_weights.to(value_heads.dtype)

        if attention_weight_bias is not None:
            attention_weights = F.softmax(
                attention_weights + attention_weight_bias, dim=-1
            )
        else:
            attention_weights = F.softmax(attention_weights, dim=-1)

        value_out = torch.einsum(
            "...htT, ...Thd->...thd", attention_weights, value_heads
        )
        value_out = value_out.reshape((*value_out.shape[:-2], -1))
        embeddings = self.output(value_out)

        return {"attention_weights": attention_weights, "embeddings": embeddings}


class PerceiverResamplerBlock(nn.Module):
    """Two cross-attentions (one per stream), then an FFN; all pre-LN + residual."""

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        ffn_embed_dim: int,
        key_size: Optional[int] = None,
        add_bias_kv: bool = False,
        add_bias_ffn: bool = True,
        ffn_activation_name: str = "gelu",
        use_glu_in_ffn: bool = False,
    ):
        super().__init__()

        if key_size is None:
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"Embedding dimension {embed_dim} should be divisible by "
                    f"num_heads {num_heads}."
                )
            key_size = embed_dim // num_heads

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim * 2 if use_glu_in_ffn else ffn_embed_dim
        self.use_glu_in_ffn = use_glu_in_ffn

        self.cross_attention_1 = MultiHeadAttention(
            num_heads=num_heads, key_size=key_size, add_bias_kv=add_bias_kv
        )
        self.cross_attention_2 = MultiHeadAttention(
            num_heads=num_heads, key_size=key_size, add_bias_kv=add_bias_kv
        )

        self.norm_cross_attention_1 = nn.LayerNorm(embed_dim)
        self.norm_cross_attention_2 = nn.LayerNorm(embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, self.ffn_embed_dim, bias=add_bias_ffn)
        self.fc2 = nn.Linear(self.ffn_embed_dim, embed_dim, bias=add_bias_ffn)

        self.activation_fn = getattr(
            nn.functional, ffn_activation_name, nn.functional.gelu
        )

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_mlp(x)
        if self.use_glu_in_ffn:
            x1, x2 = torch.chunk(self.fc1(x), 2, dim=-1)
            x = self.activation_fn(x1) * x2
        else:
            x = self.activation_fn(self.fc1(x))
        return self.fc2(x)

    def forward(
        self,
        x: torch.Tensor,
        cross_attention_embeddings_1: torch.Tensor,
        cross_attention_embeddings_2: torch.Tensor,
        attention_mask_1: Optional[torch.Tensor] = None,
        attention_mask_2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        res = x
        x = self.norm_cross_attention_1(x)
        attn_output = self.cross_attention_1(
            query=x,
            key=cross_attention_embeddings_1,
            value=cross_attention_embeddings_1,
            attention_mask=attention_mask_1,
        )["embeddings"]
        x = res + attn_output

        res = x
        x = self.norm_cross_attention_2(x)
        attn_output = self.cross_attention_2(
            query=x,
            key=cross_attention_embeddings_2,
            value=cross_attention_embeddings_2,
            attention_mask=attention_mask_2,
        )["embeddings"]
        x = res + attn_output

        x = x + self.mlp(x)

        return {"embeddings": x}


class PerceiverResampler(nn.Module):
    """Latent queries cross-attending over two streams, `num_layers` deep."""

    def __init__(self, config: PerceiverResamplerConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                PerceiverResamplerBlock(
                    num_heads=config.attention_heads,
                    embed_dim=config.embed_dim,
                    key_size=config.key_size,
                    ffn_embed_dim=config.ffn_embed_dim,
                    add_bias_kv=config.add_bias_kv,
                    add_bias_ffn=config.add_bias_ffn,
                    ffn_activation_name=config.ffn_activation_name,
                    use_glu_in_ffn=config.use_glu_in_ffn,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.latent_queries = nn.Parameter(
            torch.randn(config.resampled_length, config.embed_dim)
            * (1.0 / torch.sqrt(torch.tensor(config.embed_dim, dtype=torch.float32)))
        )

    def apply_attention_blocks(
        self,
        x: torch.Tensor,
        xf_1: torch.Tensor,
        xf_2: torch.Tensor,
        attention_mask_1: Optional[torch.Tensor] = None,
        attention_mask_2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            concat_input_1 = torch.cat([xf_1, x], dim=1)
            concat_input_2 = torch.cat([xf_2, x], dim=1)
            x = layer(
                x=x,
                cross_attention_embeddings_1=concat_input_1,
                cross_attention_embeddings_2=concat_input_2,
                attention_mask_1=attention_mask_1,
                attention_mask_2=attention_mask_2,
            )["embeddings"]
        return x

    def forward(
        self,
        input_embeddings_1: torch.Tensor,
        input_embeddings_2: torch.Tensor,
        attention_mask_1: Optional[torch.Tensor] = None,
        attention_mask_2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        assert input_embeddings_1.shape[-1] == self.config.embed_dim, (
            "stream 1 embedding dim should match embed_dim"
        )
        assert input_embeddings_2.shape[-1] == self.config.embed_dim, (
            "stream 2 embedding dim should match embed_dim"
        )

        batch_size = input_embeddings_1.shape[0]
        x = self.latent_queries.unsqueeze(0).repeat(batch_size, 1, 1)

        x = self.apply_attention_blocks(
            x=x,
            xf_1=input_embeddings_1,
            xf_2=input_embeddings_2,
            attention_mask_1=attention_mask_1,
            attention_mask_2=attention_mask_2,
        )
        return {"embeddings": x}


def build_perceiver_padding_attention_mask(
    valid_mask: torch.Tensor, resampled_length: int
) -> torch.Tensor:
    """(B, seq_len) validity mask -> (B, 1, resampled_length, seq_len+resampled_length).

    `valid_mask` is True where a stream position is real (attendable), False for
    padding. The reference derives it as `tokens != pad_token_id`; the ECG stream
    is continuous, so we take the boolean mask directly. `resampled_length` ones
    are appended for the latents, which are always attendable.
    """
    batch_size, _ = valid_mask.shape
    padding_mask = torch.cat(
        [
            valid_mask,
            torch.ones(
                (batch_size, resampled_length),
                dtype=torch.bool,
                device=valid_mask.device,
            ),
        ],
        dim=1,
    )
    padding_mask = padding_mask[:, None, None, :]
    padding_mask = padding_mask.repeat(1, 1, resampled_length, 1)
    return padding_mask


class Resampler(nn.Module):
    """Top-level module: `bio_projection` (768 -> embed_dim) + Perceiver resampler.

    forward(ecg_embeddings, task_embeddings, ecg_mask=None, task_mask=None)
        ecg_embeddings  : (B, S_ecg, input_embed_dim)   cached ECG-FM output
        task_embeddings : (B, S_task, embed_dim)        frozen-LLM task tokens
        ecg_mask        : (B, S_ecg)   True = real. None => all real.
        task_mask       : (B, S_task)  True = real. None => all real.
      -> (B, resampled_length, embed_dim) soft prompt, ready for inputs_embeds.

    Stream 1 is the ECG (bio) stream, stream 2 the task (english) stream, matching
    the reference ordering.
    """

    def __init__(self, config: PerceiverResamplerConfig):
        super().__init__()
        self.config = config
        self.input_norm = (
            nn.LayerNorm(config.input_embed_dim) if config.input_layer_norm else None
        )
        self.bio_projection = nn.Linear(config.input_embed_dim, config.embed_dim)
        self.perceiver_resampler = PerceiverResampler(config)
        self.output_norm = (
            nn.LayerNorm(config.embed_dim) if config.final_output_norm else None
        )
        if self.output_norm is not None and config.output_target_norm is not None:
            self.calibrate_output_norm(config.output_target_norm)

    def calibrate_output_norm(self, target_norm: float) -> float:
        """Set the final LayerNorm's gain so the latents come out at ``target_norm``.

        LayerNorm standardises each position to zero mean and unit variance, so the
        normalised vector already has L2 norm ~sqrt(embed_dim) before the gain is
        applied. A constant gain ``w`` scales that directly, hence
        ``w = target_norm / sqrt(embed_dim)``.

        This is initialisation, not a constraint: the gain is an ordinary trainable
        parameter and training is free to move it. Calling it on a resampler whose
        weights came from a checkpoint would overwrite what was learned, so call it
        on a freshly built model only.
        """
        if self.output_norm is None:
            raise ValueError("final_output_norm is off; there is no output norm to "
                             "calibrate")
        gain = float(target_norm) / math.sqrt(self.config.embed_dim)
        with torch.no_grad():
            self.output_norm.weight.fill_(gain)
            self.output_norm.bias.zero_()
        self.config.output_target_norm = float(target_norm)
        return gain

    def calibrate_from_embeddings(self, embedding_weight: torch.Tensor) -> float:
        """Calibrate straight from the frozen student's embedding matrix."""
        return self.calibrate_output_norm(mean_embedding_norm(embedding_weight))

    def forward(
        self,
        ecg_embeddings: torch.Tensor,
        task_embeddings: torch.Tensor,
        ecg_mask: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.input_norm is not None:
            ecg_embeddings = self.input_norm(ecg_embeddings)
        projected_ecg = self.bio_projection(ecg_embeddings)

        if ecg_mask is None:
            ecg_mask = torch.ones(
                ecg_embeddings.shape[:2], dtype=torch.bool, device=ecg_embeddings.device
            )
        if task_mask is None:
            task_mask = torch.ones(
                task_embeddings.shape[:2],
                dtype=torch.bool,
                device=task_embeddings.device,
            )

        rl = self.config.resampled_length
        ecg_attention_mask = build_perceiver_padding_attention_mask(ecg_mask.bool(), rl)
        task_attention_mask = build_perceiver_padding_attention_mask(task_mask.bool(), rl)

        latents = self.perceiver_resampler(
            input_embeddings_1=projected_ecg,
            input_embeddings_2=task_embeddings,
            attention_mask_1=ecg_attention_mask,
            attention_mask_2=task_attention_mask,
        )["embeddings"]
        if self.output_norm is not None:
            latents = self.output_norm(latents)
        return latents

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self) -> Dict[str, int]:
        """Trainable parameter count grouped by component."""

        def count(m) -> int:
            if isinstance(m, nn.Parameter):
                return m.numel()
            return sum(p.numel() for p in m.parameters())

        groups: Dict[str, int] = {}
        if self.input_norm is not None:
            groups["input_norm"] = count(self.input_norm)
        groups["bio_projection"] = count(self.bio_projection)
        groups["latent_queries"] = count(self.perceiver_resampler.latent_queries)

        ca1 = ca2 = norms = fc1 = fc2 = 0
        for layer in self.perceiver_resampler.layers:
            ca1 += count(layer.cross_attention_1)
            ca2 += count(layer.cross_attention_2)
            norms += (
                count(layer.norm_cross_attention_1)
                + count(layer.norm_cross_attention_2)
                + count(layer.norm_mlp)
            )
            fc1 += count(layer.fc1)
            fc2 += count(layer.fc2)
        groups["layers.cross_attention_1"] = ca1
        groups["layers.cross_attention_2"] = ca2
        groups["layers.norms"] = norms
        groups["layers.fc1"] = fc1
        groups["layers.fc2"] = fc2
        if self.output_norm is not None:
            groups["output_norm"] = count(self.output_norm)

        groups["total"] = sum(groups.values())
        return groups


def _report() -> None:
    for name, cfg in [
        ("ECG   embed_dim=5120", PerceiverResamplerConfig.ecg()),
        ("debug embed_dim=896 ", PerceiverResamplerConfig.debug()),
    ]:
        # Build on the meta device: counts params without allocating storage.
        with torch.device("meta"):
            model = Resampler(cfg)
        bd = model.param_breakdown()
        print(f"\n### {name}")
        width = max(len(k) for k in bd)
        for k, v in bd.items():
            if k == "total":
                print(f"    {'':{width}}   {'-' * 14}")
            print(f"    {k:<{width}} : {v:>14,}")


if __name__ == "__main__":
    _report()
