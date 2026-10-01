"""Behavioural contract for the multi-modal Perceiver resampler.

The behavioural tests run on CPU with the `debug` config (embed_dim=896); the
exact-parameter-count test uses the real ECG config (embed_dim=5120) built on the
`meta` device, so it costs no memory. The frozen student embedding layer is stood
in for by a plain `nn.Embedding` with `requires_grad=False`: asserting "the frozen
model gets no gradient" is exactly asserting that embedding gets none.

Runnable either way:
    pytest tests/test_resampler.py
    python  tests/test_resampler.py
"""

import math
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.model.resampler import (
    PerceiverResamplerConfig,
    Resampler,
    build_perceiver_padding_attention_mask,
    mean_embedding_norm,
)

torch.manual_seed(0)

B, T, N_ECG = 3, 7, 312
# The reference stack, with neither norm this repo adds; plus 2 x input_embed_dim
# for the input LayerNorm and 2 x embed_dim for the output one (both on by default).
ECG_TRAINABLE_PARAMS_REFERENCE = 843_320_320
ECG_TRAINABLE_PARAMS = ECG_TRAINABLE_PARAMS_REFERENCE + 2 * 768 + 2 * 5120

# What the ECG-FM cache actually holds: per-position norms ~350 (see the module
# docstring). The tests feed inputs at that scale rather than the unit-scale randn
# that would make the normalisation look unnecessary.
ECG_CACHE_NORM = 350.0


def _cache_scale_ecg(batch: int, dim: int, norm: float = ECG_CACHE_NORM) -> torch.Tensor:
    """ECG input at the cache's real scale: per-position L2 norm ~350."""
    x = torch.randn(batch, N_ECG, dim)
    return x * (norm / x.norm(dim=-1, keepdim=True))


def _frozen_student_embedding(embed_dim: int, vocab: int = 200) -> torch.nn.Embedding:
    emb = torch.nn.Embedding(vocab, embed_dim)
    emb.weight.requires_grad_(False)   # the student LLM stays frozen
    return emb


def _batch(cfg: PerceiverResamplerConfig, emb: torch.nn.Embedding):
    ecg = _cache_scale_ecg(B, cfg.input_embed_dim)
    task_ids = torch.randint(0, emb.num_embeddings, (B, T))
    task_embeds = emb(task_ids)                       # (B, T, embed_dim), frozen
    task_mask = torch.ones(B, T, dtype=torch.bool)
    task_mask[0, T - 2:] = False                      # pad the tail of record 0
    return ecg, task_embeds, task_mask


def test_exact_trainable_param_count_ecg_config():
    cfg = PerceiverResamplerConfig.ecg()
    with torch.device("meta"):
        model = Resampler(cfg)
    assert model.n_trainable() == ECG_TRAINABLE_PARAMS, model.n_trainable()
    # And the breakdown sums to the same number.
    bd = model.param_breakdown()
    assert bd["total"] == ECG_TRAINABLE_PARAMS
    assert sum(v for k, v in bd.items() if k != "total") == ECG_TRAINABLE_PARAMS
    assert bd["output_norm"] == 2 * cfg.embed_dim
    assert bd["input_norm"] == 2 * cfg.input_embed_dim


def test_param_count_without_output_norm_matches_the_reference():
    """final_output_norm=False must reproduce the reference stack exactly."""
    cfg = PerceiverResamplerConfig.ecg(final_output_norm=False, input_layer_norm=False)
    with torch.device("meta"):
        model = Resampler(cfg)
    assert model.output_norm is None and model.input_norm is None
    assert model.n_trainable() == ECG_TRAINABLE_PARAMS_REFERENCE, model.n_trainable()
    bd = model.param_breakdown()
    assert "output_norm" not in bd and "input_norm" not in bd


def test_output_shape():
    cfg = PerceiverResamplerConfig.debug()
    model = Resampler(cfg)
    emb = _frozen_student_embedding(cfg.embed_dim)
    ecg, task_embeds, task_mask = _batch(cfg, emb)

    out = model(ecg, task_embeds, task_mask=task_mask)
    assert out.shape == (B, cfg.resampled_length, cfg.embed_dim), tuple(out.shape)
    assert torch.isfinite(out).all()


def test_grad_reaches_all_trainable_and_no_frozen():
    cfg = PerceiverResamplerConfig.debug()
    model = Resampler(cfg)
    emb = _frozen_student_embedding(cfg.embed_dim)
    ecg, task_embeds, task_mask = _batch(cfg, emb)

    model(ecg, task_embeds, task_mask=task_mask).pow(2).mean().backward()

    for name, p in model.named_parameters():
        assert p.requires_grad, name
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
        assert p.grad.abs().sum() > 0, f"zero grad for {name}"

    assert emb.weight.grad is None


def test_batch_independence():
    """A record's output must not depend on its batch neighbours.

    Run the whole batch, then run record 1 alone, and require an identical
    result. Record 0 carries task padding, a good adversary: if the mask or the
    attention leaked across the batch this would drift.
    """
    cfg = PerceiverResamplerConfig.debug()
    model = Resampler(cfg).eval()
    emb = _frozen_student_embedding(cfg.embed_dim)
    ecg, task_embeds, task_mask = _batch(cfg, emb)

    with torch.no_grad():
        full = model(ecg, task_embeds, task_mask=task_mask)
        solo = model(ecg[1:2], task_embeds[1:2], task_mask=task_mask[1:2])

    assert torch.allclose(full[1:2], solo, atol=1e-6), (
        (full[1:2] - solo).abs().max().item()
    )


def test_input_norm_rescales_the_cache_scale_ecg_input():
    """350-norm input in, ~sqrt(input_embed_dim) out: no 262x contraction left
    for bio_projection to learn."""
    cfg = PerceiverResamplerConfig.debug()
    model = Resampler(cfg).eval()
    ecg = _cache_scale_ecg(2, cfg.input_embed_dim)
    assert abs(float(ecg.norm(dim=-1).mean()) - ECG_CACHE_NORM) < 1e-2

    with torch.no_grad():
        normed = model.input_norm(ecg)
    got = float(normed.norm(dim=-1).mean())
    expected = math.sqrt(cfg.input_embed_dim)      # unit variance per component
    assert abs(got - expected) / expected < 0.05, (got, expected)
    # and that is a ~25x reduction in what bio_projection has to absorb
    assert got < ECG_CACHE_NORM / 10


def test_input_norm_off_leaves_the_input_untouched():
    """The flag is a real switch: off, bio_projection sees the raw cache scale."""
    torch.manual_seed(11)
    off = Resampler(PerceiverResamplerConfig.debug(input_layer_norm=False)).eval()
    torch.manual_seed(11)
    on = Resampler(PerceiverResamplerConfig.debug()).eval()
    assert off.input_norm is None

    emb = _frozen_student_embedding(off.config.embed_dim)
    ecg, task_embeds, task_mask = _batch(off.config, emb)
    with torch.no_grad():
        raw = off(ecg, task_embeds, task_mask=task_mask)
        normed = on(ecg, task_embeds, task_mask=task_mask)
    assert not torch.allclose(raw, normed, atol=1e-4)


def test_input_norm_keeps_the_bio_stream_informative():
    """Normalising must not flatten records into each other.

    Two different ECGs at cache scale have to stay distinguishable after the norm,
    otherwise the fix to the scale would have cost the signal.
    """
    cfg = PerceiverResamplerConfig.debug()
    model = Resampler(cfg).eval()
    a = _cache_scale_ecg(1, cfg.input_embed_dim)
    b = _cache_scale_ecg(1, cfg.input_embed_dim)
    emb = _frozen_student_embedding(cfg.embed_dim)
    ids = torch.randint(0, emb.num_embeddings, (1, T))

    with torch.no_grad():
        out_a = model(a, emb(ids))
        out_b = model(b, emb(ids))
    assert not torch.allclose(out_a, out_b, atol=1e-5), "the ECG had no effect"


def test_calibrated_output_norm_matches_the_student_embedding_scale():
    """The whole point of the final norm: latents on the student's own scale.

    Calibrate from a frozen embedding matrix, run a real batch, and require the
    mean per-position L2 norm of the output within 2x of that matrix's own mean
    row norm -- in either direction, since a prefix that dwarfs the tokens is as
    wrong as one that vanishes next to them.
    """
    cfg = PerceiverResamplerConfig.debug()
    model = Resampler(cfg).eval()
    # A student embedding whose rows are deliberately NOT unit scale, so passing
    # the test cannot be an accident of the default initialisation.
    emb = _frozen_student_embedding(cfg.embed_dim)
    with torch.no_grad():
        emb.weight.mul_(0.05)
    target = mean_embedding_norm(emb.weight)

    gain = model.calibrate_from_embeddings(emb.weight)
    assert abs(gain - target / math.sqrt(cfg.embed_dim)) < 1e-9
    assert torch.allclose(model.output_norm.weight, torch.full_like(
        model.output_norm.weight, gain))

    ecg, task_embeds, task_mask = _batch(cfg, emb)
    with torch.no_grad():
        out = model(ecg, task_embeds, task_mask=task_mask)
    got = float(out.norm(dim=-1).mean())

    assert 0.5 * target <= got <= 2.0 * target, (
        f"latent norm {got:.3f} is not within 2x of the student's embedding "
        f"scale {target:.3f} (ratio {got / target:.2f}x)"
    )


def test_uncalibrated_output_norm_is_unit_scale():
    """Without a target the norm stays at its default gain: ~sqrt(embed_dim).

    This is the number the calibration exists to move, and on a 14B student it is
    roughly two orders of magnitude off the embedding scale.
    """
    cfg = PerceiverResamplerConfig.debug()
    assert cfg.output_target_norm is None
    model = Resampler(cfg).eval()
    emb = _frozen_student_embedding(cfg.embed_dim)
    ecg, task_embeds, task_mask = _batch(cfg, emb)

    with torch.no_grad():
        out = model(ecg, task_embeds, task_mask=task_mask)
    got = float(out.norm(dim=-1).mean())
    assert abs(got - math.sqrt(cfg.embed_dim)) / math.sqrt(cfg.embed_dim) < 0.1, got


def test_output_norm_off_reproduces_the_unnormalised_latents():
    """The flag is a real switch: off, the forward pass is the reference's."""
    torch.manual_seed(7)
    off = Resampler(PerceiverResamplerConfig.debug(final_output_norm=False)).eval()
    torch.manual_seed(7)
    on = Resampler(PerceiverResamplerConfig.debug()).eval()

    emb = _frozen_student_embedding(off.config.embed_dim)
    ecg, task_embeds, task_mask = _batch(off.config, emb)
    with torch.no_grad():
        raw = off(ecg, task_embeds, task_mask=task_mask)
        normed = on(ecg, task_embeds, task_mask=task_mask)

    # Same seed, same blocks: the only difference is the LayerNorm on top.
    with torch.no_grad():
        assert torch.allclose(on.output_norm(raw), normed, atol=1e-6)
    assert not torch.allclose(raw, normed, atol=1e-5)


def test_calibration_requires_the_norm_to_exist():
    model = Resampler(PerceiverResamplerConfig.debug(final_output_norm=False))
    try:
        model.calibrate_output_norm(1.0)
    except ValueError as err:
        assert "final_output_norm" in str(err)
    else:
        raise AssertionError("calibrating without an output norm must raise")


def test_config_target_norm_is_applied_at_construction():
    cfg = PerceiverResamplerConfig.debug(output_target_norm=3.5)
    model = Resampler(cfg).eval()
    expected = 3.5 / math.sqrt(cfg.embed_dim)
    assert torch.allclose(model.output_norm.weight,
                          torch.full_like(model.output_norm.weight, expected))


def test_mean_embedding_norm_is_the_mean_row_norm():
    w = torch.zeros(4, 9)
    w[0, 0], w[1, 0], w[2, 0], w[3, 0] = 1.0, 2.0, 3.0, 4.0
    assert abs(mean_embedding_norm(w) - 2.5) < 1e-9
    # bf16 in, fp32 arithmetic out
    assert abs(mean_embedding_norm(w.to(torch.bfloat16)) - 2.5) < 1e-2


def test_different_task_prompts_change_latents():
    """Same ECG, two different task prompts -> two different soft prompts."""
    cfg = PerceiverResamplerConfig.debug()
    model = Resampler(cfg).eval()
    emb = _frozen_student_embedding(cfg.embed_dim)

    ecg = torch.randn(1, N_ECG, cfg.input_embed_dim)
    ids_a = torch.randint(0, emb.num_embeddings, (1, T))
    ids_b = torch.randint(0, emb.num_embeddings, (1, T))
    while torch.equal(ids_a, ids_b):
        ids_b = torch.randint(0, emb.num_embeddings, (1, T))

    with torch.no_grad():
        out_a = model(ecg, emb(ids_a))
        out_b = model(ecg, emb(ids_b))

    assert not torch.allclose(out_a, out_b, atol=1e-5), "task prompt had no effect"


# --------------------------------------------------------------------------- #
# Mask semantics                                                              #
# --------------------------------------------------------------------------- #
# These prove the convention numerically rather than reading it off the source.
# The claim under test: a False entry in ecg_mask / task_mask means "this position
# is padding, never attend to it". If that is what the code does, then whatever is
# written into those positions cannot reach the output, so overwriting them has to
# leave the latents BIT-identical -- not close, identical. A flipped convention, a
# mask that never reaches the attention, or an off-by-N in the key axis all show up
# as a changed output.
#
# The junk is large but FINITE on purpose. Masked logits are set to -1e30 and
# softmaxed to exactly 0.0, and 0.0 * finite == 0.0; 0.0 * inf would be nan and the
# test would fail for arithmetic reasons rather than masking ones.
MASK_JUNK = 1e3


def _junk_like(x: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(x) * MASK_JUNK


def test_task_mask_false_positions_cannot_reach_the_output():
    """True = attend, False = padding: rewriting padding changes nothing, exactly."""
    cfg = PerceiverResamplerConfig.debug()
    model = Resampler(cfg).eval()
    emb = _frozen_student_embedding(cfg.embed_dim)
    ecg, task_embeds, task_mask = _batch(cfg, emb)
    assert not task_mask.all(), "this test needs at least one masked position"

    corrupted = task_embeds.clone()
    corrupted[~task_mask] = _junk_like(corrupted[~task_mask])

    with torch.no_grad():
        clean_out = model(ecg, task_embeds, task_mask=task_mask)
        junk_out = model(ecg, corrupted, task_mask=task_mask)

    assert torch.equal(clean_out, junk_out), (
        "task_mask=False positions changed the output, so False does NOT mean "
        "'ignore' here: the convention is inverted, or the mask never reaches the "
        f"attention. max|delta| = {(clean_out - junk_out).abs().max():.3e}")


def test_ecg_mask_false_positions_cannot_reach_the_output():
    """The same contract on the bio stream, downstream of input_norm + projection."""
    cfg = PerceiverResamplerConfig.debug()
    model = Resampler(cfg).eval()
    emb = _frozen_student_embedding(cfg.embed_dim)
    ecg, task_embeds, _ = _batch(cfg, emb)

    ecg_mask = torch.ones(B, N_ECG, dtype=torch.bool)
    ecg_mask[:, N_ECG // 2:] = False            # the back half is "padding"

    corrupted = ecg.clone()
    corrupted[~ecg_mask] = _junk_like(corrupted[~ecg_mask])

    with torch.no_grad():
        clean_out = model(ecg, task_embeds, ecg_mask=ecg_mask)
        junk_out = model(corrupted, task_embeds, ecg_mask=ecg_mask)

    assert torch.equal(clean_out, junk_out), (
        "ecg_mask=False positions changed the output; same diagnosis as the task "
        f"stream. max|delta| = {(clean_out - junk_out).abs().max():.3e}")


def test_masked_out_positions_would_otherwise_matter():
    """The control for the two tests above: an all-True mask hides nothing.

    Without this, a mask that excluded EVERYTHING (or an attention that ignored the
    stream entirely) would pass them both -- rewriting excluded positions would
    change nothing because nothing was ever read.
    """
    cfg = PerceiverResamplerConfig.debug()
    model = Resampler(cfg).eval()
    emb = _frozen_student_embedding(cfg.embed_dim)
    ecg, task_embeds, task_mask = _batch(cfg, emb)

    all_real = torch.ones_like(task_mask)
    corrupted = task_embeds.clone()
    corrupted[~task_mask] = _junk_like(corrupted[~task_mask])   # same positions

    with torch.no_grad():
        clean_out = model(ecg, task_embeds, task_mask=all_real)
        junk_out = model(ecg, corrupted, task_mask=all_real)

    assert not torch.equal(clean_out, junk_out), (
        "with an all-True mask the same positions still had no effect: the task "
        "stream is not being read at all, which would make the masking tests "
        "vacuous")


def test_padding_mask_covers_the_latent_positions():
    """The key axis is stream_len + resampled_length, and the mask spans all of it.

    The KV of each cross-attention is torch.cat([stream, latents], dim=1), so a mask
    built over stream_len alone would be short by resampled_length. Assert the built
    shape and that the appended block is True (latents always attendable).
    """
    cfg = PerceiverResamplerConfig.debug()
    seq_len, rl = 11, cfg.resampled_length
    valid = torch.ones(B, seq_len, dtype=torch.bool)
    valid[0, -3:] = False

    mask = build_perceiver_padding_attention_mask(valid, rl)

    assert mask.shape == (B, 1, rl, seq_len + rl), (
        f"expected the key axis to cover stream+latents ({seq_len + rl}), got "
        f"{tuple(mask.shape)}")
    assert mask[..., seq_len:].all(), "the latent block must be attendable"
    assert torch.equal(mask[0, 0, 0, :seq_len], valid[0]), (
        "the stream block must reproduce the validity mask position for position")


def test_the_latent_key_block_is_load_bearing():
    """Masking the appended latent columns changes the output, so they are read.

    Proves the trailing resampled_length columns are not vestigial padding that
    happens to be ignored: flip them to False and the latents stop attending to
    themselves, which must move the result.
    """
    cfg = PerceiverResamplerConfig.debug()
    model = Resampler(cfg).eval()
    emb = _frozen_student_embedding(cfg.embed_dim)
    ecg, task_embeds, _ = _batch(cfg, emb)
    rl = cfg.resampled_length

    ecg_mask = torch.ones(B, N_ECG, dtype=torch.bool)
    task_mask = torch.ones(B, T, dtype=torch.bool)
    full_ecg = build_perceiver_padding_attention_mask(ecg_mask, rl)
    full_task = build_perceiver_padding_attention_mask(task_mask, rl)
    no_latents_ecg = full_ecg.clone()
    no_latents_task = full_task.clone()
    no_latents_ecg[..., N_ECG:] = False
    no_latents_task[..., T:] = False

    with torch.no_grad():
        projected = model.bio_projection(
            model.input_norm(ecg) if model.input_norm is not None else ecg)
        with_latents = model.perceiver_resampler(
            input_embeddings_1=projected, input_embeddings_2=task_embeds,
            attention_mask_1=full_ecg, attention_mask_2=full_task)["embeddings"]
        without = model.perceiver_resampler(
            input_embeddings_1=projected, input_embeddings_2=task_embeds,
            attention_mask_1=no_latents_ecg,
            attention_mask_2=no_latents_task)["embeddings"]

    assert not torch.allclose(with_latents, without, atol=1e-6), (
        "masking the trailing latent columns changed nothing: the mask is not "
        "actually spanning the concatenated key axis")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
