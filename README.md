# ECG → LLM alignment

Extending the [ChatHealthAI](https://arxiv.org/abs/2606.02802) framework from
coded clinical data to continuous physiological signals: an ECG foundation model
feeding a frozen LLM through a task-aware resampler.

Undergraduate research with Prof. Yue Li's lab, McGill School of Computer Science.

---

## The idea

ChatHealthAI aligns structured EHR representations from a pretrained EHR
foundation model (CLMBR-T-Base) with the semantic space of a frozen LLM
(Deepseek-R1-Distill-Qwen-14B) through a task-aware resampler.

CLMBR operates on medical codes and **cannot ingest a waveform at all**. So the
extension is not "same task, different encoder" but "the framework currently
reasons over codes; here it reasons over dense physiological signal."

```
PTB-XL ECG ──► ECG-FM encoder ──► projection ──► resampler ──► frozen LLM ──► text
  (12, 2500)      (156, 768)        (156, 896)     (32, 896)
                    FROZEN          TRAINABLE      TRAINABLE      FROZEN
```

## Status

| Stage | State |
|---|---|
| PTB-XL loading + ECG-FM preprocessing | done |
| ECG-FM encoder, `encoder_out` extraction | done |
| Projection + resampler, shapes verified | done |
| End-to-end generation | done |
| **Aligner training** | **not started** |
| Calibration | later |

The aligner is currently **randomly initialised**, so the soft prompt carries no
usable information. That is the expected pre-training state, not a bug.

## Baseline result

Four different ECG segments, untrained aligner, no labels in the prompt:

```
[seg 0] true=['NORM'] -> I apologize, but I'm not able to analyze or interpret the specific ECG...
[seg 1] true=['NORM'] -> I apologize, but I'm not able to understand the context of your question...
[seg 2] true=['NORM'] -> I apologize, but I don't have access to the specific information...
[seg 3] true=['NORM'] -> I apologize, but I don't have access to the specific information...
```

Segments 2 and 3 are character-identical despite being different waveforms. The
embedding is not reaching the model in any readable form. This is the number
training has to move.

**Why the earlier "working" version was misleading:** serialising ECG-FM's 17
label predictions into the prompt text produced fluent clinical summaries, but
the LLM was only paraphrasing words it had been handed. The embedding was
redundant, so the comparison could not distinguish a working alignment from a
broken one. Same confound as evaluating a retrieval system without a closed-book
control.

## Findings worth keeping

- **Insertion point matters.** Prepending the soft prompt *before* the chat
  template opening collapses generation into repeated tokens. Inserting it
  *inside* the user turn works. The template starts with `<|im_start|>system`;
  arbitrary vectors before that break the model's expectations.
- **PTB-XL lead names.** Headers use `AVR/AVL/AVF`; ECG-FM expects
  `aVR/aVL/aVF`. `ReorderLeads(missing_lead_strategy='raise')` throws otherwise.
- **Checkpoint choice.** `mimic_iv_ecg_finetuned` exposes `encoder_out` directly
  via `build_model_from_checkpoint`; the pretrained checkpoint requires
  conversion to finetuning format first.
- **Subsampling factors measured across three encoders:** 16× (ECG-FM conv
  feature encoder), 4× (TimelyGPT CTS path), 1× (TimelyGPT ISTS/PheCode path,
  where the subsampler sits outside the wired forward path).
- **TimelyGPT was ruled out** as the encoder: no pretrained checkpoint exists
  publicly (repo is 964K, nothing on HuggingFace), and its PopHR configuration
  is 7.5M params over 315 diagnosis-only PheCodes.

## Next: training the aligner

Distillation with a privileged teacher.

1. **gpt-oss 120B** receives the ground-truth labels and writes a target summary.
2. **Student** receives only the ECG embedding and a generic question
   (`"Anything wrong with this ECG?"`) and must produce that summary.
3. **Loss:** next-token cross-entropy over the target, prompt tokens masked.
4. Gradients flow back *through* the frozen LLM into the ~3.9M trainable
   parameters of the projection and resampler.

The frozen LLM never updates but still passes gradients, which is what lets the
resampler learn to produce vectors the LLM can actually read.

Expect worse raw prediction than ECG-FM + a linear probe. The point is
language-based reasoning and interpretability, not AUROC.

## Layout

```
src/
  data.py       PTB-XL loading, lead-name fix, ECG-FM transforms
  encoder.py    ECG-FM checkpoint handling, encoder_out extraction
  modules.py    Projection + Resampler (the trainable bridge)
  generate.py   Prompt assembly, soft-prompt insertion, generation
  train.py      Distillation training loop (skeleton)
scripts/
  run_baseline.py    reproduces the three-condition baseline
notebooks/
  00_timelygpt_exploration.ipynb   earlier TimelyGPT/PTB-XL work
```

## Install

```bash
# fairseq-signals (ECG-FM's framework)
git clone https://github.com/Jwoo5/fairseq-signals
cd fairseq-signals && pip install -e . && cd ..

# ECG-FM (for label definitions and reference notebooks)
git clone https://github.com/bowang-lab/ECG-FM.git

pip install -r requirements.txt
```

PTB-XL v1.0.3 is open access. The 100 Hz records are enough to explore, but
**ECG-FM needs records500** (5 s segments at 500 Hz = 2500 samples):

```bash
wget -r -N -np -nH --cut-dirs=3 -A "*.dat,*.hea" -R "index.html*" \
  https://physionet.org/files/ptb-xl/1.0.3/records500/00000/
wget -nH --cut-dirs=3 \
  https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv \
  https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv
```

## Sources

- ECG-FM — https://github.com/bowang-lab/ECG-FM · [JAMIA Open 2025](https://doi.org/10.1093/jamiaopen/ooaf122)
- fairseq-signals — https://github.com/Jwoo5/fairseq-signals
- Checkpoints — https://huggingface.co/wanglab/ecg-fm
- ChatHealthAI — https://arxiv.org/abs/2606.02802
- PTB-XL — https://physionet.org/content/ptb-xl/1.0.3/
