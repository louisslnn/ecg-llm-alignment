"""PTB-XL loading and ECG-FM preprocessing.

PTB-XL: https://physionet.org/content/ptb-xl/1.0.3/
Preprocessing follows ECG-FM's own `ecg-transform` pipeline so that inputs
match what the encoder was trained on.
"""

import ast
import os
from typing import Dict, List, Optional

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


def _candidate_stems(fn: str, folder: str, ptb_root: str) -> List[str]:
    """Both possible on-disk stems for a PTB-XL ``filename_*`` entry.

    Handles both directory layouts: files may sit at ``<root>/records500/<sub>/``
    or flattened at ``<root>/<sub>/`` depending on the wget --cut-dirs used.
    """
    stem = os.path.splitext(fn)[0]
    flat = stem.replace(folder, "")
    return [os.path.join(ptb_root, stem), os.path.join(ptb_root, flat)]


def _resolve_stem(fn: str, folder: str, ptb_root: str) -> Optional[str]:
    """Return the on-disk stem whose ``.dat`` exists, else None."""
    for cand in _candidate_stems(fn, folder, ptb_root):
        if os.path.exists(cand + ".dat"):
            return cand
    return None


def _filename_map(ptb_root: str, rate: str = "hr") -> Dict[int, str]:
    """One CSV read -> ``{ecg_id: filename}`` for the requested rate."""
    col = "filename_hr" if rate == "hr" else "filename_lr"
    df = pd.read_csv(
        os.path.join(ptb_root, "ptbxl_database.csv"),
        index_col="ecg_id",
        usecols=["ecg_id", col],
    )
    return df[col].to_dict()


def find_missing_records(ptb_root: str, ecg_ids, rate: str = "hr") -> List[int]:
    """ecg_ids lacking BOTH a ``.dat`` and a ``.hea`` on disk, in input order.

    A single CSV read plus a couple of ``os.path.exists`` calls per record, so a
    full 21k-record working set resolves in seconds rather than minutes. Does not
    stop at the first miss -- the caller gets the complete list.
    """
    folder = "records500/" if rate == "hr" else "records100/"
    fnmap = _filename_map(ptb_root, rate)

    missing = []
    for eid in ecg_ids:
        fn = fnmap.get(eid)
        ok = False
        if fn is not None:
            for cand in _candidate_stems(fn, folder, ptb_root):
                if os.path.exists(cand + ".dat") and os.path.exists(cand + ".hea"):
                    ok = True
                    break
        if not ok:
            missing.append(eid)
    return missing


def resolve_local(df: pd.DataFrame, ptb_root: str, rate: str = "hr") -> pd.DataFrame:
    """Keep only rows whose waveform files exist on disk."""
    col = "filename_hr" if rate == "hr" else "filename_lr"
    folder = "records500/" if rate == "hr" else "records100/"

    df = df.copy()
    df["path"] = df[col].apply(lambda fn: _resolve_stem(fn, folder, ptb_root))
    out = df[df.path.notna()].copy()
    out = out[out.superclass.apply(len) > 0].copy()  # drop label-less records
    return out


def resolve_record_path(ptb_root: str, ecg_id: int, rate: str = "hr") -> str:
    """Resolve a single PTB-XL ``ecg_id`` to its on-disk WFDB path stem.

    Reads only the relevant filename column from ptbxl_database.csv. Raises
    FileNotFoundError if the waveform is not present on disk.
    """
    col = "filename_hr" if rate == "hr" else "filename_lr"
    folder = "records500/" if rate == "hr" else "records100/"

    df = pd.read_csv(
        os.path.join(ptb_root, "ptbxl_database.csv"),
        index_col="ecg_id",
        usecols=["ecg_id", col],
    )
    if ecg_id not in df.index:
        raise KeyError(f"ecg_id {ecg_id} not found in ptbxl_database.csv")

    path = _resolve_stem(df.loc[ecg_id, col], folder, ptb_root)
    if path is None:
        raise FileNotFoundError(
            f"No waveform (.dat) on disk for ecg_id {ecg_id} "
            f"({df.loc[ecg_id, col]}) under {ptb_root}"
        )
    return path


def build_schema_and_transforms(segment: bool = True):
    """ECG-FM preprocessing schema and transforms.

    ``segment=True`` (default) appends ``SegmentNonoverlapping`` so a 10 s record
    becomes 5 s / ``N_SAMPLES``-sample segments — the path the linear-probe
    baseline consumes. ``segment=False`` omits only that step, so the full record
    passes through as a single ``(12, num_samples)`` array.

    Every other step (lead reorder, resample, constant-lead handling, Standardize
    normalization) is identical on both paths; the segmentation split is the only
    difference.
    """
    schema = ECGInputSchema(
        sample_rate=SAMPLE_RATE,
        expected_lead_order=ECG_FM_LEAD_ORDER,
        required_num_samples=N_SAMPLES if segment else None,
    )
    transforms = [
        ReorderLeads(expected_order=ECG_FM_LEAD_ORDER, missing_lead_strategy="raise"),
        LinearResample(desired_sample_rate=SAMPLE_RATE),
        HandleConstantLeads(strategy="zero"),
        Standardize(),
    ]
    if segment:
        transforms.append(SegmentNonoverlapping(segment_length=N_SAMPLES))
    return schema, transforms


def preprocess_record(path: str, schema, transforms):
    """Load a WFDB record and run the given ECG-FM transforms.

    Returns ``(source, ECGInput)`` where ``source`` is a float32 tensor of shape
    ``(n_segments, 12, N_SAMPLES)`` when segmenting or ``(1, 12, num_samples)``
    when not. Shared by ``PTBXLDataset`` and the full-record encoder path so the
    preprocessing is defined exactly once.
    """
    sig, meta = wfdb.rdsamp(path)
    feats = sig.T.astype(np.float32)  # (leads, samples)
    md = ECGMetadata(
        sample_rate=meta["fs"],
        num_samples=feats.shape[1],
        lead_names=fix_leads(meta["sig_name"]),
        unit=None,
        input_start=0,
        input_end=feats.shape[1],
    )
    md.file = path
    inp = ECGInput(feats, md)
    sample = ECGSample(inp, schema, transforms)
    return torch.from_numpy(sample.out).float(), inp


class PTBXLDataset(Dataset):
    """Yields (source, ECGInput).

    With ``segment=True`` (default) ``source`` is (n_seg, 12, 2500): a 10 s
    PTB-XL record at 500 Hz produces 2 non-overlapping 5 s segments. With
    ``segment=False`` it is (1, 12, num_samples): the full record, unsegmented.
    """

    def __init__(self, frame: pd.DataFrame, segment: bool = True):
        self.paths = list(frame.path)
        self.labels = list(frame.superclass)
        self.schema, self.transforms = build_schema_and_transforms(segment)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return preprocess_record(self.paths[i], self.schema, self.transforms)


def collate(batch):
    """Flatten per-record segments into one batch dimension."""
    sources = torch.cat([b[0] for b in batch])
    inps = [b[1] for b in batch]
    return sources, inps


def make_loader(frame: pd.DataFrame, batch_size: int = 4, num_workers: int = 0,
                segment: bool = True):
    return DataLoader(
        PTBXLDataset(frame, segment=segment),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )
