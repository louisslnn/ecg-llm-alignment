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
