"""Full-record encoder path: shape, dtype and finiteness on real PTB-XL records.

Runs `src.encoder.encode_full_record` on 3 records and checks that each returns
a (312, 768) fp32 tensor with no NaN/Inf. Needs the finetuned checkpoint (auto-
downloaded on first use) and records500 waveforms on disk.

Runnable either way:
    pytest tests/test_encode_full_record.py
    python  tests/test_encode_full_record.py
"""

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import data as D
from src import encoder as E

PTB_ROOT = os.environ.get("PTB_ROOT", os.path.join(REPO_ROOT, "data/ptbxl"))
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(REPO_ROOT, "data/ckpts"))


def _available_ecg_ids(n=3):
    """First `n` ecg_ids whose waveforms are present on disk."""
    frame = D.resolve_local(D.load_metadata(PTB_ROOT), PTB_ROOT)
    assert len(frame) >= n, f"need >= {n} local PTB-XL records, found {len(frame)}"
    return list(frame.index[:n])


def test_encode_full_record_shape_dtype_finite():
    ckpt = E.download_checkpoint(CKPT_DIR)
    model = E.load_encoder(ckpt, device="cpu")

    ecg_ids = _available_ecg_ids(3)
    assert len(ecg_ids) == 3

    for ecg_id in ecg_ids:
        emb = E.encode_full_record(model, ecg_id, PTB_ROOT, device="cpu")

        assert emb.shape == (E.ENCODER_TOKENS, E.ENCODER_DIM), (ecg_id, tuple(emb.shape))
        assert emb.dtype == torch.float32, (ecg_id, emb.dtype)
        assert torch.isfinite(emb).all(), f"non-finite values for ecg_id {ecg_id}"


if __name__ == "__main__":
    test_encode_full_record_shape_dtype_finite()
    print("ok: encode_full_record shape/dtype/finite on 3 records")
