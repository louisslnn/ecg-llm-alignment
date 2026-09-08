# ECG → LLM alignment

Extending the [ChatHealthAI](https://arxiv.org/abs/2606.02802) framework from coded
clinical data to continuous physiological signals: an ECG foundation model feeding a
frozen LLM through a resampler.

Undergraduate research in Prof. Yue Li's lab, McGill School of Computer Science.

---

## Idea

ChatHealthAI aligns structured EHR representations from a pretrained EHR foundation
model (CLMBR-T-Base) with the semantic space of a frozen LLM through a task-aware
resampler.

CLMBR operates on medical codes and **cannot ingest a waveform at all**. So this is
not "same task, different encoder" but "the framework currently reasons over codes;
here it reasons over dense physiological signal."

```
PTB-XL ECG ──► ECG-FM encoder ──► projection ──► resampler ──► frozen LLM ──► text
 (12, 2500)       (156, 768)        (156, 896)     (32, 896)
                    FROZEN          TRAINABLE      TRAINABLE       FROZEN
```

Measured chain: `(8, 12, 2500) → (8, 156, 768) → (8, 32, 896)`
Trainable parameters: **3,934,336** (projection 689k, resampler 3.2M)

## Status

| Stage | State |
|---|---|
| PTB-XL loading + ECG-FM preprocessing | done |
| ECG-FM encoder, `encoder_out` extraction | done |
| Projection + resampler, shapes verified | done |
| End-to-end generation, baseline documented | done |
| **Aligner training** | **not started** |
| Calibration | later |

The aligner is **randomly initialised**. That is the expected pre-training state,
not a bug: training it is the next task.

## Baseline result

`scripts/run_baseline.py` runs three conditions on the same ECG segments.

| | prompt contains | result |
|---|---|---|
| **A** | text only | honest refusal: *"I don't have access to an embedded representation..."* |
| **B** | soft prompt, no labels | fluent but confabulated |
| **C** | soft prompt + labels as text | unstable; sometimes empty, sometimes echoes the labels |

All eight segments in `results/baseline.json` are ground-truth **NORM**. Not one
condition-B output identifies normal sinus rhythm. Instead:

```
"You are the authority on this site."
"You are the final boss?"
"The ECG you've described is not a standard 12-lead ECG ... likely created by
 clipping together parts of multiple leads from different patients."
"...could indicate that the ECG has been corrupted or tampered with."
```

**Reading:** the soft prompt does perturb generation (a random linear map preserves
some structure, so different ECGs give different outputs), but nothing survives in a
form the LLM can interpret. Worse than useless: condition A produces an *honest
refusal*, while condition B converts that into *confident fabrication*, including
unfounded claims of data corruption.

That is what alignment training has to fix, and it connects directly to the
calibration question.

**Why an earlier version of this experiment was misleading:** serialising ECG-FM's 17
label predictions into the prompt text produced fluent clinical summaries, but the
LLM was only paraphrasing words it had been handed. The embedding was redundant, so
the comparison could not distinguish a working alignment from a broken one. Same
confound as evaluating a retrieval system with no closed-book control.

## Findings worth keeping

- **Insertion point matters.** Prepending the soft prompt *before* the chat template
  opening collapses generation into repeated tokens. Inserting it *inside* the user
  turn works. The template begins `<|im_start|>system`; arbitrary vectors before that
  break the model's expectations.
- **PTB-XL lead names.** Headers use `AVR/AVL/AVF`; ECG-FM expects `aVR/aVL/aVF`.
  `ReorderLeads(missing_lead_strategy='raise')` throws otherwise. See `LEAD_FIX`.
- **Checkpoint choice.** `mimic_iv_ecg_finetuned` exposes `encoder_out` directly via
  `build_model_from_checkpoint`; the pretrained checkpoint needs conversion to
  finetuning format first (see ECG-FM's `infer_cli.ipynb`).
- **Subsampling factors across three encoders:** 16× (ECG-FM conv feature encoder),
  4× (TimelyGPT CTS path), 1× (TimelyGPT ISTS/PheCode path, where the subsampler sits
  outside the wired forward path).
- **TimelyGPT ruled out** as the encoder: no pretrained checkpoint exists publicly
  (repo is 964K; nothing on HuggingFace), and its PopHR configuration is 7.5M params
  over 315 diagnosis-only PheCodes against CLMBR-T-Base's 141M over 65,536 codes.
- **Results are seed-dependent.** A different random initialisation gives different
  garbage. On one seed all eight outputs were byte-identical refusals; on another they
  varied. The content is the finding, not the diversity count.

## Next: training the aligner

Distillation with a privileged teacher.

1. **gpt-oss 120B** receives the ground-truth labels and writes a target summary.
2. **Student** receives only the ECG embedding and a generic question
   (`"Anything wrong with this ECG?"`) and must produce that summary.
3. **Loss:** next-token cross-entropy over the target; prompt tokens masked to `-100`.
4. Gradients flow *through* the frozen LLM into the 3.9M trainable parameters.

The frozen LLM never updates but still passes gradients, which is what lets the
resampler learn to emit vectors the LLM can read.

Expect worse raw prediction than ECG-FM + a linear probe. The point is language-based
reasoning and interpretability, not AUROC.

**Open questions** (see the checklist at the top of `src/train.py`): what the teacher
sees, per-record or per-segment targets, whether the encoder stays frozen, which
student LLM, and available compute.

## Layout

```
src/
  data.py       PTB-XL loading, lead-name fix, ECG-FM transforms
  encoder.py    ECG-FM checkpoint handling, encoder_out extraction
  modules.py    Projection + Resampler (the trainable bridge)
  generate.py   Prompt assembly, soft-prompt insertion, generation
  train.py      Distillation training loop (skeleton)
scripts/
  run_baseline.py    three-condition baseline; --stratify samples one per superclass
results/
  baseline.json      committed output
```

## Install

Needs **Python 3.11**. fairseq-signals asks for ≤3.9 and breaks on 3.14 (argparse
rejects `str | None` as a `type=` callable). On macOS 26, Homebrew Python 3.11 also
has a broken `pyexpat`; Miniforge avoids both problems.

```bash
conda create -n ecgfm python=3.11 -y && conda activate ecgfm

git clone https://github.com/Jwoo5/fairseq-signals
git clone https://github.com/bowang-lab/ECG-FM.git

pip install -r requirements.txt
pip install pip==24.0
cd fairseq-signals && pip install -e . && cd ..
pip install --upgrade pip
```

PTB-XL v1.0.3 is open access. **records500 is required** (5 s at 500 Hz = 2500 samples):

```bash
mkdir -p data/ptbxl && cd data/ptbxl
wget -r -N -np -nH --cut-dirs=3 -A "*.dat,*.hea" -R "index.html*" \
  https://physionet.org/files/ptb-xl/1.0.3/records500/00000/
wget -nH --cut-dirs=3 \
  https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv \
  https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv
```

## Run

```bash
mkdir -p results
python scripts/run_baseline.py \
  --ptb ./data/ptbxl --ckpts ./data/ckpts --ecgfm-repo ./ECG-FM \
  --device cpu --n-records 4 --out results/baseline.json

# one record per diagnostic superclass (NORM, MI, STTC, CD, HYP)
python scripts/run_baseline.py \
  --ptb ./data/ptbxl --ckpts ./data/ckpts --ecgfm-repo ./ECG-FM \
  --device cpu --stratify --out results/baseline_stratified.json
```

The ECG-FM checkpoint (~1 GB) downloads from HuggingFace on first run.

## Sources

- ECG-FM — https://github.com/bowang-lab/ECG-FM · [JAMIA Open 2025](https://doi.org/10.1093/jamiaopen/ooaf122)
- fairseq-signals — https://github.com/Jwoo5/fairseq-signals
- Checkpoints — https://huggingface.co/wanglab/ecg-fm
- ChatHealthAI — https://arxiv.org/abs/2606.02802
- PTB-XL — https://physionet.org/content/ptb-xl/1.0.3/