#!/usr/bin/env python
"""How much the resampler's 64 latents move with the ECG, against with the question.

Given the overfit_final.pt the gate left behind, run its resampler over the examples
the gate trained on and measure two spreads of the output:

    ACROSS RECORDS   the same superclass question, the different ECGs.
    WITHIN A RECORD  the same ECG, its five different superclass questions.

The contrast is the measurement. The first is variation the waveform causes, the
second is variation the task stream causes. If across-records is a small fraction of
within-record, the resampler is keyed on the prompt and is ignoring the ECG. That is
a different failure from a prefix the language model declines to read, and it points
at a different fix: the first is inside the resampler, the second is at the splice.

RUNS ON A LOGIN NODE: CPU, no GPU, no full model load. The task stream needs the
student's token embeddings and nothing else of the student, so only the
embed_tokens tensor is read out of the safetensors (~1.5 GB of bf16), not the 14B.
The resampler's weights are memory-mapped from the checkpoint rather than copied.
Everything the run needs is in the checkpoint's payload: the architecture flags,
which examples the gate drew (overfit_selection; the first --overfit N for a
checkpoint that predates --overfit-select), the task stream, strip_think, the model
dir. The prompts are built here, so the checkpoint is put through
evaluate.require_strip_think_match with that strip_think before anything else loads.

TWO ACROSS-RECORDS ROWS, because "the same question" is not the same task stream.
The student prompt is a paraphrase drawn per (ecg_id, superclass) plus the patient's
age and sex (src/data/dataset.py student_prompt), so two records asked about the
same superclass are handed different text. Comparing the gate's own inputs across
records therefore moves the ECG AND the prompt, and a resampler that reads nothing
but the prompt still shows a spread there. So:

    prompt held fixed  one record's prompt for that superclass, byte-identical,
                       served with each record's ECG in turn. Only the waveform
                       differs: this is the row the contrast is read from.
    own prompts        each record under its own prompt, exactly the inputs the
                       gate trained on. What the student actually received; the gap
                       to the row above is what the wording, age and sex add.

Held-fixed pairs an ECG with another record's prompt, which the resampler was not
trained on -- the same thing the gate's shuffled control does, from the other side.

THE NUMBERS. Every comparison is between two (positions, dim) tensors and is taken
per position, then averaged over positions and over pairs:

    cosine dist     1 - cos(a_k, b_k). Scale-free already.
    L2 / mean norm  ||a_k - b_k|| over the mean per-position norm of the tensors
                    being compared, so it reads as a fraction of a latent's length.

The INPUT row is the same two numbers on the cached ECG-FM embeddings themselves,
(312, 768), across the same records. It is the scale to judge the output against:
what the resampler was handed, next to what it made of it.

FP32. The gate ran the resampler under bf16 autocast; this runs it in fp32, so the
numbers are the resampler's function, a little more finely than the student saw it.
bf16 resolves about 2^-8 (0.4%) per component: read a spread near that with care.

    python scripts/latent_spread.py --ckpt $CKPT_DIR/overfit_final.pt

One resampler forward per example plus one per (question, record) for the held-fixed
row: 95 for --overfit 50, a few seconds each on CPU.
"""

import argparse
import glob
import itertools
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import (  # noqa: E402
    DatasetConfig,
    build_datasets,
    make_collate_fn,
    prompt_only_task_mask,
)
from src.model.resampler import PerceiverResamplerConfig, Resampler  # noqa: E402
from src.teacher.prompt import SUPERCLASSES  # noqa: E402
from evaluate import require_strip_think_match  # noqa: E402

EMBED_KEY_SUFFIX = "embed_tokens.weight"
LINE_W = 92


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="the gate's overfit_final.pt")
    # The gate's own paths are in the payload; these are for a checkpoint that has
    # been moved to a machine where they do not exist.
    ap.add_argument("--model-dir", default=None,
                    help="student checkpoint dir, for the tokenizer and the "
                         "embed_tokens tensor (default: the one the gate ran with)")
    ap.add_argument("--emb-cache", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--teacher", default=None)
    return ap.parse_args()


def load_payload(path):
    """The checkpoint, memory-mapped, and the arguments of the run that wrote it."""
    # our own checkpoint (trusted); it carries numpy/python RNG state that the
    # weights_only=True default (torch>=2.6) refuses to unpickle.
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    run = payload.get("args") if isinstance(payload, dict) else None
    if not run or run.get("overfit") is None:
        sys.exit(f"{path} is not an overfit_final.pt: its payload has no --overfit "
                 "run arguments, so there is no saying which examples it trained on")
    return payload, run


def mapped_resampler(payload, run):
    """The resampler the gate judged, built from the run's own architecture flags.

    Built on the meta device and handed the memory-mapped tensors (assign=True), so
    the 843M weights are paged in from the file instead of being initialised and
    then overwritten: 3.4 GB that never has to sit in a login node's memory twice.
    """
    with torch.device("meta"):
        resampler = Resampler(PerceiverResamplerConfig.ecg(
            input_layer_norm=not run["no_input_norm"],
            final_output_norm=not run["no_output_norm"]))
    resampler.load_state_dict(payload["resampler"], strict=True, assign=True)
    return resampler.eval()


def load_embed_tokens(model_dir):
    """The student's input-embedding matrix, and nothing else of the student.

    safe_open reads a shard's header only; get_tensor then pulls the one tensor. The
    transformer blocks are never touched.
    """
    from safetensors import safe_open

    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    hits = []
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            hits += [(shard, k) for k in f.keys() if k.endswith(EMBED_KEY_SUFFIX)]
    if len(hits) != 1:
        sys.exit(f"expected one *{EMBED_KEY_SUFFIX} tensor in the {len(shards)} "
                 f"safetensors file(s) under {model_dir}, found {len(hits)}: {hits}")
    shard, key = hits[0]
    with safe_open(shard, framework="pt") as f:
        weight = f.get_tensor(key)
    print(f"token embeddings: {key} {tuple(weight.shape)} {weight.dtype} from "
          f"{os.path.basename(shard)} (the only tensor read from the student)")
    return weight


def gate_indices(dataset, payload, run):
    """Which train examples the gate drew, read from the checkpoint, not re-derived.

    ``overfit_selection`` is train.py's own record of its draw. Under stratified the
    examples are every example of the recorded records; under head, and for a
    checkpoint written before the record existed, they are the first --overfit N.
    Fatal if the records found here are not the ones recorded: the manifest or
    teacher file is then not the run's, and the latents would be of other examples.
    """
    examples = dataset.examples
    selection = payload.get("overfit_selection")
    if selection and selection["select"] == "stratified":
        chosen = set(selection["ecg_ids"])
        indices = [i for i, e in enumerate(examples) if int(e.ecg_id) in chosen]
    else:
        indices = list(range(min(run["overfit"], len(examples))))
    found = list(dict.fromkeys(int(examples[i].ecg_id) for i in indices))
    if selection and (found != selection["ecg_ids"] or len(indices) != run["overfit"]):
        sys.exit(f"the checkpoint records {run['overfit']} examples from records "
                 f"{selection['ecg_ids']}; this dataset gives {len(indices)} from "
                 f"{found}. Not the manifest and teacher file the gate ran on")
    print("gate examples: " + (f"--overfit-select {selection['select']}, as recorded "
                               "in the checkpoint" if selection else
                               f"the first {len(indices)} train examples (the "
                               "checkpoint predates --overfit-select)"))
    return indices


def gate_examples(dataset, indices, run, tokenizer, embed):
    """The examples the gate trained on, as the resampler's two input streams.

    Each goes through training's own collate and task mask, one at a time, and the
    stream is cut to the positions the mask keeps: a masked key has attention weight
    exactly zero, so dropping it is the same forward without the CPU time.

    Returns ``(ecg, task, prompts)``: ecg_id -> (312, 768), and
    (ecg_id, superclass) -> the (T, embed_dim) fp32 stream and the prompt's text.
    """
    collate = make_collate_fn(tokenizer, max_length=run["max_length"],
                              strip_think=dataset.config.strip_think)
    ecg, task, prompts = {}, {}, {}
    for i in indices:
        item = dataset[i]
        batch = collate([item])
        attn, labels = batch["attention_mask"], batch["labels"]
        mask = (prompt_only_task_mask(attn, labels) if run["task_stream"] == "prompt"
                else attn.bool())
        key = (int(item["ecg_id"]), item["superclass"])
        ecg[key[0]] = item["ecg_embed"]
        task[key] = embed[batch["input_ids"][0][mask[0]]].float()
        prompts[key] = item["task_prompt"]
    return ecg, task, prompts


@torch.no_grad()
def run_resampler(resampler, ecg, task, jobs):
    """(ecg of, prompt of, superclass) -> (64, embed_dim) latents, one forward each."""
    out, t0 = {}, time.time()
    for n, (ecg_of, prompt_of, sc) in enumerate(jobs, 1):
        out[(ecg_of, prompt_of, sc)] = resampler(
            ecg[ecg_of][None], task_embeddings=task[(prompt_of, sc)][None])[0]
        if n == 1 or n % 10 == 0 or n == len(jobs):
            print(f"  forward {n:>3}/{len(jobs)}  {time.time() - t0:6.0f}s", flush=True)
    return out


def mean_norm(tensors):
    """Mean per-position L2 norm over a list of (positions, dim) tensors."""
    return float(torch.stack(tensors).double().norm(dim=-1).mean())


def pairwise(tensors):
    """Every pair of (positions, dim) tensors: [(cosine distance, L2)], per position
    and then averaged over positions.

    In fp64, on the difference itself: the case this script exists for is tensors
    that barely differ, and that is where 1 - cos and a Gram-matrix L2 cancel away.
    Identical vectors still leave 1 - cos at -1e-16 or so, hence the clamp.
    """
    x = torch.stack(tensors).double()
    return [(float((1.0 - F.cosine_similarity(x[i], x[j], dim=-1)).clamp_min(0).mean()),
             float((x[i] - x[j]).norm(dim=-1).mean()))
            for i, j in itertools.combinations(range(len(x)), 2)]


def spread(groups, norm):
    """Mean pairwise spread, pairs taken within each group and pooled over groups.

    A group of fewer than two has no pair and adds nothing. Pure on the tensors.
    """
    pairs = [p for g in groups for p in pairwise(g)] if groups else []
    if not pairs:
        return {"pairs": 0, "cos": None, "l2": None, "norm": norm}
    cos, l2 = np.mean(pairs, axis=0)
    return {"pairs": len(pairs), "cos": float(cos), "l2": float(l2) / norm,
            "norm": norm}


def _num(v):
    return f"{'n/a':>12}" if v is None else f"{v:>12.4g}"


def _row(label, s):
    return (f"{label:<40} {s['pairs']:>6} {_num(s['cos'])} {_num(s['l2'])} "
            f"{s['norm']:>12.4g}")


def _ratio(label, top, bottom):
    def r(k):
        return None if top[k] is None or not bottom[k] else top[k] / bottom[k]
    print(f"{label:<47} {_num(r('cos'))} {_num(r('l2'))}")


def main():
    args = parse_args()
    payload, run = load_payload(args.ckpt)
    verdict = {True: "PASSED", False: "FAILED"}.get(payload.get("gate_passed"), "n/a")
    print(f"checkpoint {args.ckpt}: step {payload.get('global_step')}, gate {verdict}, "
          f"--overfit {run['overfit']}, task_stream={run['task_stream']}, "
          f"strip_think={run['strip_think']}")
    # The prompts below are built with the payload's own strip_think, so this can
    # only refuse a payload whose args and meta disagree; there is no way past it.
    require_strip_think_match(args.ckpt, run["strip_think"], escape_flag=None)
    resampler = mapped_resampler(payload, run)

    model_dir = args.model_dir or run["model_dir"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    embed = load_embed_tokens(model_dir)
    if embed.shape[1] != resampler.config.embed_dim:
        sys.exit(f"{model_dir} embeds tokens at {embed.shape[1]}, the resampler runs "
                 f"at {resampler.config.embed_dim}: not the student it trained against")

    cfg_kwargs = {"strip_think": bool(run["strip_think"])}
    for name, field in (("emb_cache", "emb_cache_dir"), ("manifest", "manifest_path"),
                        ("teacher", "teacher_path")):
        path = getattr(args, name) or run.get(name)
        if path:
            cfg_kwargs[field] = path
    dataset = build_datasets(DatasetConfig(**cfg_kwargs))["train"]
    ecg, task, prompts = gate_examples(dataset, gate_indices(dataset, payload, run),
                                       run, tokenizer, embed)
    del embed

    records = list(ecg)                     # dataset order, as the gate saw them
    asked = {sc: [r for r in records if (r, sc) in task] for sc in SUPERCLASSES}
    asked = {sc: rs for sc, rs in asked.items() if rs}
    print(f"{len(task)} examples: {len(records)} records x {len(asked)} questions"
          + ("" if len(task) == len(records) * len(asked) else
             " (incomplete grid: a record missing a question joins no pair for it)"))
    if len(records) < 2:
        sys.exit("fewer than two records: there is no other ECG to compare against")

    # The prompt held fixed for each question is the first record's that has one.
    anchor = {sc: rs[0] for sc, rs in asked.items()}
    print("prompt held fixed, per question:")
    for sc, r in anchor.items():
        print(f"  {sc:<5} record {r:>6}: {prompts[(r, sc)]!r}")

    own = [(r, r, sc) for r, sc in task]
    held = [(r, anchor[sc], sc) for sc in asked for r in records]
    jobs = list(dict.fromkeys(own + held))  # the anchor's own example is in both
    print(f"\n{len(jobs)} resampler forwards on CPU ({len(own)} gate examples + "
          f"{len(jobs) - len(own)} with the prompt held fixed):", flush=True)
    lat = run_resampler(resampler, ecg, task, jobs)

    out_norm = mean_norm([lat[j] for j in own])
    inputs = [ecg[r] for r in records]
    rows = {
        "input": spread([inputs], mean_norm(inputs)),
        "held": spread([[lat[(r, anchor[sc], sc)] for r in records] for sc in asked],
                       out_norm),
        "own": spread([[lat[(r, r, sc)] for r in rs] for sc, rs in asked.items()],
                      out_norm),
        "within": spread([[lat[(r, r, sc)] for sc in asked if (r, sc) in task]
                          for r in records], out_norm),
    }
    in_shape = "x".join(str(n) for n in inputs[0].shape)
    out_shape = "x".join(str(n) for n in lat[own[0]].shape)

    print("\n" + "=" * LINE_W)
    print(f"LATENT SPREAD over the {len(task)} examples the gate trained on "
          f"(mean pairwise, per position)")
    print("=" * LINE_W)
    print(f"{'':<40} {'pairs':>6} {'cosine dist':>12} {'L2 / norm':>12} "
          f"{'mean norm':>12}")
    print("-" * LINE_W)
    print(_row(f"INPUT {in_shape}, across records", rows["input"]))
    print(f"OUTPUT {out_shape}")
    print(_row("  ACROSS RECORDS, prompt held fixed", rows["held"]))
    print(_row("  across records, own prompts", rows["own"]))
    print(_row("  WITHIN A RECORD, across questions", rows["within"]))
    print("-" * LINE_W)
    _ratio("across (held fixed) / within", rows["held"], rows["within"])
    _ratio("across (held fixed), output / input", rows["held"], rows["input"])
    print("-" * LINE_W)
    print("across / within near zero => the latents follow the question, not the ECG: "
          "the resampler\n  is keyed on the prompt. Near or above one => the waveform "
          "moves them as much as the\n  question does, and a failing gate is not the "
          "resampler ignoring its input.")
    print("output / input near zero  => records that differ going in barely differ "
          "coming out.")
    print("own prompts minus held fixed is what the differing wording, age and sex "
          "add; it is not\n  the ECG.")

    print(f"\nacross records, by question\n{'':<8} {'held fixed':>32} "
          f"{'own prompts':>32}\n{'':<8} "
          + " ".join([f"{'pairs':>6} {'cosine dist':>12} {'L2 / norm':>12}"] * 2))
    for sc, rs in asked.items():
        cells = [spread([[lat[(r, anchor[sc], sc)] for r in records]], out_norm),
                 spread([[lat[(r, r, sc)] for r in rs]], out_norm)]
        print(f"{sc:<8} " + " ".join(
            f"{s['pairs']:>6} {_num(s['cos'])} {_num(s['l2'])}" for s in cells))


if __name__ == "__main__":
    main()
