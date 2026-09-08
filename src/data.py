"""PTB-XL loading and ECG-FM preprocessing.

PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
Preprocessing follows ECG-FM's own `ecg-transform` pipeline so that inputs
match what the encoder was trained on.
"""

import ast
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import DataLoader, Dataset

from ecg_transform.inp import ECGInput, ECGInputSchema
from ecg_transform.sample import ECGMetadata, ECGSample
from ecg_transform.t.common import HandleConstantLeads, LinearResample, ReorderLeads
from ecg_transform.t.cut import SegmentNonoverlapping
from ecg_transform.t.scale import Standardize

# ECG-FM expects this exact lead order.
ECG_FM_LEAD_ORDER = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# PTB-XL headers spell the augmented leads in upper case. ReorderLeads raises
# on a mismatch, so they must be renamed before the transform runs.
LEAD_FIX = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}

SAMPLE_RATE = 500
SEGMENT_SECONDS = 5
N_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS  # 2500

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def fix_leads(names: List[str]) -> List[str]:
    """Normalise PTB-XL lead names to ECG-FM's expected spelling."""
    return [LEAD_FIX.get(n, n) for n in names]


def load_metadata(ptb_root: str) -> pd.DataFrame:
    """Load ptbxl_database.csv and derive diagnostic superclasses."""
    df = pd.read_csv(os.path.join(ptb_root, "ptbxl_database.csv"), index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    agg = pd.read_csv(os.path.join(ptb_root, "scp_statements.csv"), index_col=0)
    agg = agg[agg.diagnostic == 1]

    def to_superclass(codes):
        out = set()
        for k in codes:
            if k in agg.index and pd.notna(agg.loc[k].diagnostic_class):
                out.add(agg.loc[k].diagnostic_class)
        return sorted(out)

    df["superclass"] = df.scp_codes.apply(to_superclass)
    return df


def resolve_local(df: pd.DataFrame, ptb_root: str, rate: str = "hr") -> pd.DataFrame:
    """Keep only rows whose waveform files exist on disk.

    Handles both directory layouts: files may sit at ``<root>/records500/<sub>/``
    or flattened at ``<root>/<sub>/`` depending on the wget --cut-dirs used.
    """
    col = "filename_hr" if rate == "hr" else "filename_lr"
    folder = "records500/" if rate == "hr" else "records100/"

    def resolve(fn: str) -> Optional[str]:
        stem = os.path.splitext(fn)[0]
        flat = stem.replace(folder, "")
        for cand in (os.path.join(ptb_root, stem), os.path.join(ptb_root, flat)):
            if os.path.exists(cand + ".dat"):
                return cand
        return None

    df = df.copy()
    df["path"] = df[col].apply(resolve)
    out = df[df.path.notna()].copy()
    out = out[out.superclass.apply(len) > 0].copy()  # drop label-less records
    return out


def build_schema_and_transforms():
    schema = ECGInputSchema(
        sample_rate=SAMPLE_RATE,
        expected_lead_order=ECG_FM_LEAD_ORDER,
        required_num_samples=N_SAMPLES,
    )
    transforms = [
        ReorderLeads(expected_order=ECG_FM_LEAD_ORDER, missing_lead_strategy="raise"),
        LinearResample(desired_sample_rate=SAMPLE_RATE),
        HandleConstantLeads(strategy="zero"),
        Standardize(),
        SegmentNonoverlapping(segment_length=N_SAMPLES),
    ]
    return schema, transforms


class PTBXLDataset(Dataset):
    """Yields (segments, ECGInput) where segments is (n_seg, 12, 2500).

    A 10 s PTB-XL record at 500 Hz produces 2 non-overlapping 5 s segments.
    """

    def __init__(self, frame: pd.DataFrame):
        self.paths = list(frame.path)
        self.labels = list(frame.superclass)
        self.schema, self.transforms = build_schema_and_transforms()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        sig, meta = wfdb.rdsamp(self.paths[i])
        feats = sig.T.astype(np.float32)  # (leads, samples)
        md = ECGMetadata(
            sample_rate=meta["fs"],
            num_samples=feats.shape[1],
            lead_names=fix_leads(meta["sig_name"]),
            unit=None,
            input_start=0,
            input_end=feats.shape[1],
        )
        md.file = self.paths[i]
        inp = ECGInput(feats, md)
        sample = ECGSample(inp, self.schema, self.transforms)
        return torch.from_numpy(sample.out).float(), inp


def collate(batch):
    """Flatten per-record segments into one batch dimension."""
    sources = torch.cat([b[0] for b in batch])
    inps = [b[1] for b in batch]
    return sources, inps


def make_loader(frame: pd.DataFrame, batch_size: int = 4, num_workers: int = 0):
    return DataLoader(
        PTBXLDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )
