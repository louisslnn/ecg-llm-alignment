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

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.model.resampler import PerceiverResamplerConfig, Resampler

torch.manual_seed(0)

B, T, N_ECG = 3, 7, 312
ECG_TRAINABLE_PARAMS = 843_320_320


def _frozen_student_embedding(embed_dim: int, vocab: int = 200) -> torch.nn.Embedding:
    emb = torch.nn.Embedding(vocab, embed_dim)
    emb.weight.requires_grad_(False)   # the student LLM stays frozen
    return emb


def _batch(cfg: PerceiverResamplerConfig, emb: torch.nn.Embedding):
    ecg = torch.randn(B, N_ECG, cfg.input_embed_dim)
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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
