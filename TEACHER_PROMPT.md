# Teacher Prompt v1 — ECG Reasoning Supervision Generation

Adapted from ChatHealthAI A.1.2. Teacher: gpt-oss 120B, run offline once over the
working set, cached to disk.

**Decisions settled**

| Question | Decision |
|---|---|
| Task formulation | 5 binary questions per record, one per diagnostic superclass. Agreed with Bo Hong. |
| Evidence source | PTB-XL `report` field, translated and normalized, plus recording properties. |
| Refine stage | **None.** A.1.1 exists to compress up to 30 EHR events. Median report here is 93 chars. Translation and normalization fold into this prompt. |

---

## Preprocessing, before the teacher sees anything

Applied in code, not left to the model.

1. **Strip the autogen suffix.** Unvalidated records end with a boilerplate marker
   (`4.46 ... unbestÄtigter bericht`). It reveals annotation quality and is not
   evidence. Remove it.
2. **Strip curator edits.** Some reports contain injected label text, e.g.
   `Edit: NORM 80, (NORM 100, IVCB)` in `ecg_id` 18822. This leaks the annotation
   into the evidence channel. Remove any `Edit:` segment and any parenthesised
   code-with-likelihood pattern.
3. **Length floor.** Reports run from 1 to 397 chars. Below ~15 chars there is no
   usable content; fall back to annotations-only evidence for those records and log
   how many.
4. **Empty reports.** 2 records. Same fallback.

---

## Prompt

> **You are generating reasoning supervision for a 12-lead ECG interpretation task.**
>
> **Task:**
> - The question is: **"Does this ECG show evidence of {SUPERCLASS_NAME}?"**
> - The label is already known and must be followed. The known answer is: **{YES_NO}**
> - The goal is to generate structured reasoning supervision consistent with the
>   known label.
>
> **Input:**
> - **Clinical report:** the free-text interpretation recorded for this ECG. It may
>   be in German, English, or Swedish, and is often telegraphic. Translate to
>   English internally. It may contain both observations of the trace (rhythm,
>   axis, intervals, waveform morphology, voltages) and diagnostic conclusions.
> - **Annotated findings:** SCP-ECG statements assigned by a cardiologist, each with
>   a description, its class (diagnostic, form, rhythm), and an annotation likelihood.
> - **Recording properties:** heart axis, infarction stadium, and signal-quality
>   annotations (baseline drift, static noise, burst noise, electrode problems)
>   where present.
> - **Patient context:** age and sex.
>
> **Label semantics, binding:**
> - A likelihood of 0 means the cardiologist **stated no likelihood**, not that the
>   finding is absent. Never write a negation, denial, or "no evidence of" claim on
>   the basis of a likelihood of 0.
> - A finding not present in the list was **not annotated**. Do not assert it is
>   absent. Silence is not negation.
> - Only diagnostic statements determine the answer to the question. Rhythm and
>   form statements are additional description and do not by themselves make a
>   recording abnormal.
>
> **Evidence rules:**
> - Evidence items are **observations of the recording**, drawn from the clinical
>   report and the recording properties. Rhythm, rate, axis, intervals, waveform
>   morphology, voltages, signal quality.
> - The report may state a diagnosis. **A diagnosis is a conclusion, not evidence.**
>   Put observational content in Evidence and diagnostic content in Reasoning or
>   Conclusion.
> - Do not invent observations. If the report describes only a rhythm, Evidence
>   contains only that rhythm. Thin evidence is correct; fabricated evidence is not.
> - Do not state numeric measurements (rate, intervals, amplitudes) unless they
>   appear in the input.
>
> **Output format**
> The generated answer must follow the structure:
>
> ```
> Evidence: 1) ... 2) ...
> Reasoning:
> Step 1: Using Evidence ...
> Step 2: Using Evidence ...
> Conclusion: Yes/No, ...
> <END>
> ```
>
> **Generation rules**
> - Use only information supported by the provided input.
> - Do not hallucinate diagnoses, findings, measurements, or clinical history.
> - Each reasoning step must explicitly reference Evidence items.
> - Do not infer or mention prior events, comorbidities, medications, or management.
>   No investigations, no treatment suggestions, no follow-up advice.
> - Age and sex may be referenced as **given** patient context. Never present them
>   as inferred from the recording.
> - Signal-quality annotations may be used to qualify confidence in a finding.
> - Use cautious clinical language such as "suggests", "consistent with", or
>   "likely" when appropriate. Avoid unsupported severity claims.
> - Where a finding carries a low annotation likelihood, reflect that uncertainty in
>   the reasoning rather than stating it flatly.
> - The Conclusion must agree with the known label. Yes/No first, then a short
>   explanation.
> - At most 5 Evidence items. Whole answer under 200 words.

---

## Superclass names for `{SUPERCLASS_NAME}`

| Code | Question text |
|---|---|
| NORM | a normal ECG |
| MI | myocardial infarction |
| STTC | ST/T changes |
| CD | a conduction disturbance |
| HYP | hypertrophy |

NORM inverts: the question reads naturally as "Is this a normal ECG?" and the
Evidence/Reasoning structure still applies.

## Cost note, decide before the full run

5 questions x 21,793 records = **108,965 teacher generations**. At ~200 words each
that is a substantial run against the gpt-oss API.

Cheaper alternative: one teacher call per record emitting all five blocks in a
single response, then split in code. ~5x fewer calls, and the teacher sees the full
annotation once rather than five times. Downside is a longer response per call and
a parsing step. **Raise with Bo Hong before the full generation.**

## Pilot (1.5)

50 records, stratified across: NORM+SR only; multi-label abnormal; records with a
low-likelihood diagnostic finding; records with signal-quality annotations; German,
English and Swedish reports; validated and unvalidated; and at least one Yes and
one No for each of the 5 superclass questions.

Read all 50 by hand. What to check, in order:

1. **Negations from likelihood 0.** The single most likely failure.
2. **Diagnosis appearing as Evidence** rather than as Conclusion.
3. **Fabricated observations** not present in the report.
4. **Leaked boilerplate or curator edits** that preprocessing missed.
5. **Degenerate NORM targets** — how similar are the 7,062 {NORM, SR} outputs to
   each other.