"""ECG-FM encoder wrapper.

ECG-FM: https://github.com/bowang-lab/ECG-FM
fairseq-signals: https://github.com/Jwoo5/fairseq-signals
Checkpoints: https://huggingface.co/wanglab/ecg-fm

We use the `mimic_iv_ecg_finetuned` checkpoint rather than the pretrained one:
`build_model_from_checkpoint` exposes `encoder_out` directly, whereas the
pretrained model must first be converted to finetuning format via
`fairseq-hydra-train` (see ECG-FM's infer_cli.ipynb).
"""

import os
from typing import Dict, List, Tuple

import pandas as pd
import torch

CKPT_REPO = "wanglab/ecg-fm"
CKPT_NAME = "mimic_iv_ecg_finetuned"

# A PTB-XL records500 record is 10 s @ 500 Hz = 5000 samples. The conv feature
# encoder subsamples 16x, so a full record yields exactly 312 (= 5000 // 16)
# local tokens of width 768. These are fixed by the checkpoint architecture.
FULL_RECORD_SAMPLES = 5000
ENCODER_TOKENS = 312
ENCODER_DIM = 768


def download_checkpoint(dest_dir: str) -> str:
    """Fetch the finetuned checkpoint + config. Returns the .pt path."""
    from huggingface_hub import hf_hub_download

    os.makedirs(dest_dir, exist_ok=True)
    paths = {}
    for ext in (".pt", ".yaml"):
        fn = CKPT_NAME + ext
        dst = os.path.join(dest_dir, fn)
        if not os.path.exists(dst):
            hf_hub_download(repo_id=CKPT_REPO, filename=fn, local_dir=dest_dir)
        paths[ext] = dst
    return paths[".pt"]


def load_encoder(ckpt_path: str, device: str = "cuda"):
    """Load ECG-FM. Returns an ECGTransformerClassificationModel in eval mode."""
    from fairseq_signals.models import build_model_from_checkpoint

    model = build_model_from_checkpoint(checkpoint_path=ckpt_path)
    model.eval()
    model.to(device)
    return model


def load_label_names(ecgfm_repo_root: str) -> List[str]:
    """The 17 label names the finetuned head predicts."""
    label_def = pd.read_csv(
        os.path.join(ecgfm_repo_root, "data/mimic_iv_ecg/labels/label_def.csv"),
        index_col="name",
    )
    return label_def.index.to_list()


@torch.no_grad()
def encode(model, source: torch.Tensor, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Run ECG-FM on (batch, 12, 2500).

    Returns a dict with:
      encoder_out : (batch, ~156, 768)  local representations, PRE-pooling
      logits      : (batch, 17)         clinical label logits
    """
    out = model(source=source.to(device))
    return {"encoder_out": out["encoder_out"], "logits": out["out"]}


def load_full_record_source(ecg_id: int, ptb_root: str) -> torch.Tensor:
    """Preprocessed full record as a ``(12, 5000)`` fp32 tensor (no batch dim).

    Applies the identical ECG-FM preprocessing with segmentation OFF
    (``build_schema_and_transforms(segment=False)``). Raises ValueError if the
    record is not exactly 5000 samples (10 s @ 500 Hz); the signal is never
    padded or cropped to fit.
    """
    try:
        from . import ptbxl_data as D
    except ImportError:  # allow flat `sys.path.insert(0, "src")` imports too
        import ptbxl_data as D

    path = D.resolve_record_path(ptb_root, ecg_id)
    schema, transforms = D.build_schema_and_transforms(segment=False)
    source, _ = D.preprocess_record(path, schema, transforms)  # (1, 12, num_samples)

    source = source.squeeze(0)  # (12, num_samples)
    if tuple(source.shape) != (12, FULL_RECORD_SAMPLES):
        raise ValueError(
            f"ecg_id {ecg_id}: expected a (12, {FULL_RECORD_SAMPLES}) record "
            f"(10 s @ {D.SAMPLE_RATE} Hz) but got {tuple(source.shape)}; "
            "refusing to pad or crop."
        )
    assert tuple(source.shape) == (12, FULL_RECORD_SAMPLES)
    return source.float()


@torch.no_grad()
def _encoder_only_forward(model, batch: torch.Tensor) -> torch.Tensor:
    """Run the finetuning encoder's forward on ``(B, 12, 5000)``, return fp32.

    Bypasses the classification head (``self.proj``): we call the finetuning
    ``forward`` directly and read its output *before* the ``encoder_out_to_emb``
    average pooling. Returns ``(B, 312, 768)`` fp32 on ``batch``'s device.
    """
    from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel

    model.eval()
    res = ECGTransformerFinetuningModel.forward(model, source=batch)
    return res["x"].float()


@torch.no_grad()
def encode_full_record(model, ecg_id: int, ptb_root: str, device: str = "cpu"
                       ) -> torch.Tensor:
    """Pre-pooling encoder representation of one full PTB-XL record.

    Returns ``encoder_out`` as a ``(312, 768)`` fp32 tensor with **no** batch
    dimension. Computation is fp32; the caller casts if it wants lower precision.
    See :func:`load_full_record_source` and :func:`_encoder_only_forward` for the
    (12, 5000) input assertion and the head-free forward.
    """
    source = load_full_record_source(ecg_id, ptb_root)  # (12, 5000), asserted
    batch = source.unsqueeze(0).to(device)  # (1, 12, 5000), fp32
    encoder_out = _encoder_only_forward(model, batch).squeeze(0)  # (312, 768)

    assert tuple(encoder_out.shape) == (ENCODER_TOKENS, ENCODER_DIM), (
        f"expected encoder_out {(ENCODER_TOKENS, ENCODER_DIM)}, "
        f"got {tuple(encoder_out.shape)}"
    )
    return encoder_out.detach()


@torch.no_grad()
def encode_full_records(model, ecg_ids, ptb_root: str, device: str = "cpu"
                        ) -> torch.Tensor:
    """Batched version of :func:`encode_full_record`: one forward for N records.

    Loads each record via :func:`load_full_record_source` (per-record (12, 5000)
    assertion and raise-on-wrong-length), stacks them into a single
    ``(N, 12, 5000)`` batch, and runs one head-free encoder forward. Returns a
    fp32 ``(N, 312, 768)`` tensor on CPU, row-aligned to ``ecg_ids``.
    """
    ecg_ids = list(ecg_ids)
    sources = [load_full_record_source(eid, ptb_root) for eid in ecg_ids]  # each (12, 5000)
    batch = torch.stack(sources).to(device)  # (N, 12, 5000), fp32
    assert tuple(batch.shape) == (len(ecg_ids), 12, FULL_RECORD_SAMPLES), (
        f"expected batch {(len(ecg_ids), 12, FULL_RECORD_SAMPLES)}, got {tuple(batch.shape)}"
    )

    out = _encoder_only_forward(model, batch)  # (N, 312, 768), fp32
    assert tuple(out.shape) == (len(ecg_ids), ENCODER_TOKENS, ENCODER_DIM), (
        f"expected {(len(ecg_ids), ENCODER_TOKENS, ENCODER_DIM)}, got {tuple(out.shape)}"
    )
    return out.detach().cpu()


def pool_global(encoder_out: torch.Tensor) -> torch.Tensor:
    """ECG-FM's own average pooling (see encoder_out_to_emb in their notebook).

    NOT used in this pipeline: the resampler consumes the pre-pooled local
    representations. Kept here for reference and for linear-probe baselines.
    """
    return torch.div(encoder_out.sum(dim=1), (encoder_out != 0).sum(dim=1))


def top_labels(logits: torch.Tensor, label_names: List[str], k: int = 5
               ) -> List[List[Tuple[str, float]]]:
    """Per-sample top-k (name, probability), sorted descending."""
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return [
        sorted(zip(label_names, row), key=lambda t: -t[1])[:k]
        for row in probs
    ]
