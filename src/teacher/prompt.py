"""Render a teacher view plus one superclass question into the final prompt.

The template text lives here, once, under an explicit :data:`PROMPT_VERSION`. It
follows ``TEACHER_PROMPT.md`` (Teacher Prompt v1); keep the two in sync and bump
:data:`PROMPT_VERSION` whenever the wording changes, so every cached generation
records which prompt produced it.

The 5 superclass questions and their Yes/No answers derive from the record's
**diagnostic** statements (via :func:`superclass_answer`): a likelihood of 0 still
asserts a diagnostic class, so it counts. NORM is answered "Yes" only when the
diagnostic classes are exactly ``{NORM}`` -- the project's normality definition.
"""

from typing import Dict, List

from .view import TeacherView, build_view

# v6: (2) add a German/Swedish terminology glossary before the input section, so
# report terms like Lagetyp render as "electrical axis" rather than "lead type";
# (3) make the diagnosis-in-Evidence rule concrete with worked examples, since the
# abstract v5 rule was ignored when the report itself stated a conclusion. The
# supervision-pipeline tell that v5 left ("...yet the required answer is
# affirmative") is removed deterministically in post-processing, not the prompt
# (see src/teacher/postprocess.py). v5 banned supervision/meta language in the
# output. See TEACHER_PROMPT.md.
PROMPT_VERSION = "teacher-v6"

# Superclass -> the phrase that fills {SUPERCLASS_NAME} in the question. Order is
# fixed so a per-record five-block run is deterministic.
SUPERCLASSES: List[str] = ["NORM", "MI", "STTC", "CD", "HYP"]
SUPERCLASS_NAME: Dict[str, str] = {
    "NORM": "a normal ECG",
    "MI": "myocardial infarction",
    "STTC": "ST/T changes",
    "CD": "a conduction disturbance",
    "HYP": "hypertrophy",
}


def question_text(superclass: str) -> str:
    """The question sentence for a superclass.

    NORM asks directly ("Is this a normal ECG?"); the four abnormal superclasses
    ask whether the ECG shows evidence of the finding.
    """
    if superclass not in SUPERCLASS_NAME:
        raise ValueError(f"unknown superclass {superclass!r}")
    if superclass == "NORM":
        return "Is this a normal ECG?"
    return f"Does this ECG show evidence of {SUPERCLASS_NAME[superclass]}?"

_NO_REPORT_PLACEHOLDER = (
    "(no usable free-text report; rely on the annotated findings below)"
)

# The teacher prompt body, from TEACHER_PROMPT.md. {input_block} is rendered from
# the view; {question} and {yes_no} come from the superclass question.
PROMPT_TEMPLATE = """\
You are generating reasoning supervision for a 12-lead ECG interpretation task.

Task:
- The question is: "{question}"
- The label is already known and must be followed. The known answer is: {yes_no}
- The goal is to generate structured reasoning supervision consistent with the
  known label.

Terminology. The report may be in German or Swedish. Use these renderings:
| Source | English |
|---|---|
| Lagetyp (normal / Linkstyp / Steiltyp) | electrical axis (normal / left axis / vertical axis) |
| el-axel, vänster el-axel | electrical axis, left axis deviation |
| QRS-Achse, schwierig bestimmbare QRS-Achse | QRS axis, QRS axis difficult to determine |
| Schenkelblock, Rechtsschenkelblock, Linksschenkelblock | bundle branch block, right BBB, left BBB |
| unvollständiger | incomplete |
| skänkelblock, högersidigt / vänstersidigt | bundle branch block, right-sided / left-sided |
| Niederspannung, periphere Niederspannung | low voltage, low voltage in the limb leads |
| Linksbelastung / Rechtsbelastung | left / right ventricular strain |
| kammarhypertrofi, vänster kammarhypertrofi | ventricular hypertrophy, left ventricular hypertrophy |
| Amplitudenkriterien für Linkshypertrophie | voltage criteria for left ventricular hypertrophy |
| Myokardschaden | myocardial damage |
| p-sinistrocardiale | P mitrale, left atrial enlargement pattern |
| Erregungsrückbildungsstörung | repolarisation abnormality |
| Hebung / Senkung | elevation / depression |
| avvikande QRS(T) förlopp | abnormal QRS(T) progression |
| t-förändring | T-wave change |
| Verdacht auf | suspicion of |
| möglich / wahrscheinlich | possible / probable |
| ålder ej bestämmbar / sannolikt äldre | age not determinable / probably old |
| Grenzbefund | borderline finding |
| unauffällig | unremarkable |

Input:
- Clinical report: the free-text interpretation recorded for this ECG. It may be
  in German, English, or Swedish, and is often telegraphic. Translate to English
  internally. It may contain both observations of the trace (rhythm, axis,
  intervals, waveform morphology, voltages) and diagnostic conclusions.
- Annotated findings: SCP-ECG statements assigned by a cardiologist, each with a
  description, its class (diagnostic, form, rhythm), and an annotation likelihood.
- Recording properties: heart axis, infarction stadium, and signal-quality
  annotations (baseline drift, static noise, burst noise, electrode problems)
  where present.
- Patient context: age and sex.

Label semantics, binding:
- A likelihood of 0 means the cardiologist stated no likelihood, not that the
  finding is absent. Never write a negation, denial, or "no evidence of" claim on
  the basis of a likelihood of 0.
- A finding not present in the list was not annotated. Do not assert it is absent.
  Silence is not negation.
- Only diagnostic statements determine the answer to the question. Rhythm and form
  statements are additional description and do not by themselves make a recording
  abnormal.

Evidence rules:
- Evidence items are observations of the recording: rhythm, rate, intervals,
  waveform morphology, voltages, lead-specific findings, signal quality. They come
  from the clinical report and the signal-quality annotations.
- Every numbered Evidence item must state something the report or a signal-quality
  annotation positively describes. Never write an Evidence item asserting that
  something is not mentioned, not described, or absent from the report -- no "no
  mention of ST elevation", no "no description of bundle-branch block", no "absence
  of hypertrophy criteria". Reasoning about what the report does not cover belongs
  in the Reasoning section, never in Evidence. If there are not enough positive
  observations, list fewer Evidence items; one honest item beats three padded ones.
- Never cite the heart axis or infarction stadium recording-property fields as
  Evidence. Like the annotated findings, they are annotation fields, not
  observations of the trace, and are not visible in the recording. Among the
  recording properties, only the signal-quality annotations may appear in Evidence.
- Never cite the annotated findings as Evidence. Do not write "annotated finding",
  "diagnostic annotation", "likelihood", "the cardiologist", or any paraphrase of
  these. The annotations tell you the correct answer; they are not observations of
  the trace and must not appear in Evidence or Reasoning. This rule has no
  exceptions, including when the annotation is the only information that supports
  the known answer.
- Never cite the absence of an annotation. Do not write "no diagnostic statement
  indicates X", "not annotated", "the cardiologist did not assign X", or any
  equivalent. Absence of an annotation is not an observation.
- Quote the report, do not extend it. If the report says a wave is best seen in a
  lead, that is a statement about visibility, not amplitude. Do not convert a
  described observation into a different one because it better supports the known
  answer.
- Do not state numeric measurements (rate, intervals, amplitudes) unless they
  appear in the input.
- Reports usually end with the interpreter's own diagnostic conclusion, for
  example "consistent with ischaemic heart disease with old inferior myocardial
  infarction" or "inferior infarkt bör övervägas". That conclusion never goes in
  Evidence, in any phrasing, including "X noted" or "X described". Evidence takes
  only the observations: rhythm, axis, intervals, waveform morphology, voltages,
  lead-specific findings, signal quality. The diagnostic conclusion may be used in
  Reasoning. Example. Report: "sinus rhythm. q waves in ii, iii, avf. st segments
  are depressed in i, avl, v5,6. consistent with old inferior myocardial
  infarction." Evidence takes the sinus rhythm, the Q waves and the ST depression.
  It does not take "old inferior myocardial infarction". Reasoning may say the Q
  waves in the inferior leads indicate a prior infarction.

When there is nothing relevant to the question:
This is common and expected, and it is the correct situation for most No answers.
List whatever observations the input does describe, then state in Reasoning that
those listed observations do not include the findings the question concerns -- for
example, "the listed observations do not include QRS-voltage or chamber-enlargement
criteria". Phrase this as a statement about the recording, not about what the
report does or does not mention. Thin evidence is correct; a short honest block is
better than a padded one. Never fill the gap by citing annotations, citing their
absence, or inventing an observation.

When the known answer is Yes but nothing in the input supports the finding:
Do not construct a clinical argument to justify the answer. Do not reason from one
finding to another by typical association, and do not use age, sex, or clinical
plausibility as support. List whatever observations the input does describe, state
in Reasoning that those listed observations do not bear on the question, and assert
the finding directly in the Conclusion -- for example "Conclusion: Yes, the
recording shows hypertrophy." Do not explain any discrepancy and do not hedge about
what was or was not described.

Output format
The generated answer must follow the structure:

Evidence: 1) ... 2) ...
Reasoning:
Step 1: Using Evidence ...
Step 2: Using Evidence ...
Conclusion: Yes/No, ...
<END>

Generation rules
- Write only about the recording. The Evidence, Reasoning and Conclusion must read
  as reasoning about an ECG and nothing else. Never mention the known answer, the
  label, the correct answer, what is required, the written report, the clinical
  interpretation, or the annotations, in any phrasing. Do not write "the report
  does not describe", "not captured in the written interpretation", "the known
  answer", or equivalents -- these are facts about how the supervision was
  produced, not about the ECG.
- Use only information supported by the provided input.
- Do not hallucinate diagnoses, findings, measurements, or clinical history.
- Each reasoning step must explicitly reference Evidence items.
- Do not infer or mention prior events, comorbidities, medications, or management.
  No investigations, no treatment suggestions, no follow-up advice.
- Age and sex may be referenced as given patient context. Never present them as
  inferred from the recording.
- Signal-quality annotations may be used to qualify confidence in a finding.
- Use cautious clinical language such as "suggests", "consistent with", or
  "likely" when appropriate. Avoid unsupported severity claims.
- Where a finding carries a low annotation likelihood, reflect that uncertainty in
  the reasoning rather than stating it flatly.
- The Conclusion must agree with the known label: begin with Yes or No, then a
  short explanation. For a No answer, phrase it as "the recording as described
  shows no findings of X" rather than "the ECG does not show evidence of X". For
  the NORM question, a No instead reads "No, the recording as described is not a
  normal ECG". Keep the "Conclusion: No," prefix exactly.
- At most 5 Evidence items. Whole answer under 200 words.

---
{input_block}"""


def superclass_answer(view: TeacherView, superclass: str) -> bool:
    """Known Yes/No answer for one superclass, from the diagnostic statements.

    A diagnostic statement asserts its class regardless of likelihood (0 included).
    NORM is "Yes" only when the diagnostic classes are exactly ``{NORM}``; each
    abnormal superclass is "Yes" when it appears among the diagnostic classes.
    """
    if superclass not in SUPERCLASS_NAME:
        raise ValueError(f"unknown superclass {superclass!r}")
    diagnostic_classes = {
        s.diagnostic_superclass
        for s in view.statements
        if "diagnostic" in s.classes and s.diagnostic_superclass is not None
    }
    if superclass == "NORM":
        return diagnostic_classes == {"NORM"}
    return superclass in diagnostic_classes


def superclass_answers(view: TeacherView) -> Dict[str, bool]:
    """All five known answers, in :data:`SUPERCLASSES` order."""
    return {sc: superclass_answer(view, sc) for sc in SUPERCLASSES}


def _format_likelihood(likelihood: float) -> str:
    return str(int(likelihood)) if float(likelihood).is_integer() else str(likelihood)


def _render_input_block(view: TeacherView) -> str:
    lines: List[str] = []

    report = view.report_text if view.has_usable_report else _NO_REPORT_PLACEHOLDER
    lines.append(f"Clinical report: {report}")

    lines.append("Annotated findings:")
    if view.statements:
        for s in view.statements:
            classes = ", ".join(s.classes) if s.classes else "unclassified"
            desc = s.description or s.code
            lines.append(
                f"  - {desc} [class: {classes}; likelihood: "
                f"{_format_likelihood(s.likelihood)}]"
            )
    else:
        lines.append("  - none")

    lines.append("Recording properties:")
    lines.append(f"  Heart axis: {view.heart_axis or 'not annotated'}")
    stadium = ", ".join(view.infarction_stadium) if view.infarction_stadium else "not annotated"
    lines.append(f"  Infarction stadium: {stadium}")
    quality = "; ".join(view.signal_quality) if view.signal_quality else "none annotated"
    lines.append(f"  Signal quality: {quality}")

    age = "not recorded" if view.age is None else _format_likelihood(view.age)
    sex = view.sex or "not recorded"
    lines.append(f"Patient context: age {age}, sex {sex}")

    return "\n".join(lines)


def render_prompt(view: TeacherView, superclass: str) -> str:
    """Render the full teacher prompt for one view and one superclass question."""
    if superclass not in SUPERCLASS_NAME:
        raise ValueError(f"unknown superclass {superclass!r}")
    yes_no = "Yes" if superclass_answer(view, superclass) else "No"
    return PROMPT_TEMPLATE.format(
        question=question_text(superclass),
        yes_no=yes_no,
        input_block=_render_input_block(view),
    )


def render_all_prompts(view: TeacherView) -> Dict[str, str]:
    """Render all five superclass prompts for a view, in fixed order."""
    return {sc: render_prompt(view, sc) for sc in SUPERCLASSES}


def render_record(rec: dict, superclass: str) -> str:
    """Convenience: build a view from a manifest entry and render one prompt."""
    return render_prompt(build_view(rec), superclass)


# --- Mode B: one call per record, five blocks in one response -----------------
# The one-call-per-record design flagged in TEACHER_PROMPT.md. The shared rule
# text below mirrors PROMPT_TEMPLATE; keep the two in sync (and bump PROMPT_VERSION)
# when the rules change. The output contract differs: five superclass-headed blocks
# in a fixed order, each terminated by <END>, parsed by src/teacher/parse.py. The
# worked example is Mode-B-specific (it uses the ### CODE block shape).
MODE_B_TEMPLATE = """\
You are generating reasoning supervision for a 12-lead ECG interpretation task.

Task:
- You are given ONE ECG and FIVE yes/no questions about it, one per diagnostic
  superclass. Each answer is already known and must be followed.
- The questions and their known answers:
{questions_block}
- Produce structured reasoning supervision for each question, consistent with its
  known answer.

Terminology. The report may be in German or Swedish. Use these renderings:
| Source | English |
|---|---|
| Lagetyp (normal / Linkstyp / Steiltyp) | electrical axis (normal / left axis / vertical axis) |
| el-axel, vänster el-axel | electrical axis, left axis deviation |
| QRS-Achse, schwierig bestimmbare QRS-Achse | QRS axis, QRS axis difficult to determine |
| Schenkelblock, Rechtsschenkelblock, Linksschenkelblock | bundle branch block, right BBB, left BBB |
| unvollständiger | incomplete |
| skänkelblock, högersidigt / vänstersidigt | bundle branch block, right-sided / left-sided |
| Niederspannung, periphere Niederspannung | low voltage, low voltage in the limb leads |
| Linksbelastung / Rechtsbelastung | left / right ventricular strain |
| kammarhypertrofi, vänster kammarhypertrofi | ventricular hypertrophy, left ventricular hypertrophy |
| Amplitudenkriterien für Linkshypertrophie | voltage criteria for left ventricular hypertrophy |
| Myokardschaden | myocardial damage |
| p-sinistrocardiale | P mitrale, left atrial enlargement pattern |
| Erregungsrückbildungsstörung | repolarisation abnormality |
| Hebung / Senkung | elevation / depression |
| avvikande QRS(T) förlopp | abnormal QRS(T) progression |
| t-förändring | T-wave change |
| Verdacht auf | suspicion of |
| möglich / wahrscheinlich | possible / probable |
| ålder ej bestämmbar / sannolikt äldre | age not determinable / probably old |
| Grenzbefund | borderline finding |
| unauffällig | unremarkable |

Input:
- Clinical report: the free-text interpretation recorded for this ECG. It may be
  in German, English, or Swedish, and is often telegraphic. Translate to English
  internally. It may contain both observations of the trace (rhythm, axis,
  intervals, waveform morphology, voltages) and diagnostic conclusions.
- Annotated findings: SCP-ECG statements assigned by a cardiologist, each with a
  description, its class (diagnostic, form, rhythm), and an annotation likelihood.
- Recording properties: heart axis, infarction stadium, and signal-quality
  annotations (baseline drift, static noise, burst noise, electrode problems)
  where present.
- Patient context: age and sex.

Label semantics, binding:
- A likelihood of 0 means the cardiologist stated no likelihood, not that the
  finding is absent. Never write a negation, denial, or "no evidence of" claim on
  the basis of a likelihood of 0.
- A finding not present in the list was not annotated. Do not assert it is absent.
  Silence is not negation.
- Only diagnostic statements determine the answer to a question. Rhythm and form
  statements are additional description and do not by themselves make a recording
  abnormal.

Evidence rules:
- Evidence items are observations of the recording: rhythm, rate, intervals,
  waveform morphology, voltages, lead-specific findings, signal quality. They come
  from the clinical report and the signal-quality annotations.
- Every numbered Evidence item must state something the report or a signal-quality
  annotation positively describes. Never write an Evidence item asserting that
  something is not mentioned, not described, or absent from the report -- no "no
  mention of ST elevation", no "no description of bundle-branch block", no "absence
  of hypertrophy criteria". Reasoning about what the report does not cover belongs
  in the Reasoning section, never in Evidence. If there are not enough positive
  observations, list fewer Evidence items; one honest item beats three padded ones.
- Never cite the heart axis or infarction stadium recording-property fields as
  Evidence. Like the annotated findings, they are annotation fields, not
  observations of the trace, and are not visible in the recording. Among the
  recording properties, only the signal-quality annotations may appear in Evidence.
- Never cite the annotated findings as Evidence. Do not write "annotated finding",
  "diagnostic annotation", "likelihood", "the cardiologist", or any paraphrase of
  these. The annotations tell you the correct answer; they are not observations of
  the trace and must not appear in Evidence or Reasoning. This rule has no
  exceptions, including when the annotation is the only information that supports
  the known answer.
- Never cite the absence of an annotation. Do not write "no diagnostic statement
  indicates X", "not annotated", "the cardiologist did not assign X", or any
  equivalent. Absence of an annotation is not an observation.
- Quote the report, do not extend it. If the report says a wave is best seen in a
  lead, that is a statement about visibility, not amplitude. Do not convert a
  described observation into a different one because it better supports the known
  answer.
- Do not state numeric measurements (rate, intervals, amplitudes) unless they
  appear in the input.
- Reports usually end with the interpreter's own diagnostic conclusion, for
  example "consistent with ischaemic heart disease with old inferior myocardial
  infarction" or "inferior infarkt bör övervägas". That conclusion never goes in
  Evidence, in any phrasing, including "X noted" or "X described". Evidence takes
  only the observations: rhythm, axis, intervals, waveform morphology, voltages,
  lead-specific findings, signal quality. The diagnostic conclusion may be used in
  Reasoning. Example. Report: "sinus rhythm. q waves in ii, iii, avf. st segments
  are depressed in i, avl, v5,6. consistent with old inferior myocardial
  infarction." Evidence takes the sinus rhythm, the Q waves and the ST depression.
  It does not take "old inferior myocardial infarction". Reasoning may say the Q
  waves in the inferior leads indicate a prior infarction.

When there is nothing relevant to a question:
This is common and expected, and it is the correct situation for most No answers.
List whatever observations the input does describe, then state in Reasoning that
those listed observations do not include the findings the question concerns -- for
example, "the listed observations do not include QRS-voltage or chamber-enlargement
criteria". Phrase this as a statement about the recording, not about what the
report does or does not mention. Thin evidence is correct; a short honest block is
better than a padded one. Never fill the gap by citing annotations, citing their
absence, or inventing an observation.

When the known answer is Yes but nothing in the input supports the finding:
Do not construct a clinical argument to justify the answer. Do not reason from one
finding to another by typical association, and do not use age, sex, or clinical
plausibility as support. List whatever observations the input does describe, state
in Reasoning that those listed observations do not bear on the question, and assert
the finding directly in the Conclusion -- for example "Conclusion: Yes, the
recording shows hypertrophy." Do not explain any discrepancy and do not hedge about
what was or was not described.

Output format
Produce exactly five blocks, in this order: NORM, MI, STTC, CD, HYP. Head each
block with its superclass code alone on its own line, then Evidence, Reasoning and
a Conclusion, and terminate every block with <END>:

### NORM
Evidence: 1) ... 2) ...
Reasoning:
Step 1: Using Evidence ...
Step 2: Using Evidence ...
Conclusion: Yes/No, ...
<END>

### MI
Evidence: 1) ...
Reasoning:
Step 1: Using Evidence ...
Conclusion: Yes/No, ...
<END>

Continue with ### STTC, ### CD and ### HYP in the same format.

Example of a correct block where the report is silent on the question:

### HYP
Evidence: 1) Sinus rhythm. 2) Q waves in II, III, aVF. 3) ST depression in I, aVL, V5, V6.
Reasoning:
Step 1: Evidence 1-3 describe rhythm and inferolateral repolarisation findings.
Step 2: None of Evidence 1-3 concern QRS voltage, R-wave amplitude, or chamber enlargement, which are the findings relevant to this question, so the listed observations include no hypertrophy criteria.
Conclusion: No, the recording as described shows no findings of hypertrophy.
<END>

Example of a correct block where the known answer is Yes but the report is silent on the finding:

### HYP
Evidence: 1) Sinus rhythm. 2) Right bundle branch block. 3) P waves best seen in V1.
Reasoning:
Step 1: Evidence 1-3 describe rhythm and a conduction pattern; none concern QRS voltage, R-wave amplitude, or chamber enlargement, which are the findings relevant to this question.
Step 2: The listed observations do not bear on hypertrophy, and none of them should be reinterpreted as QRS-voltage or chamber-enlargement criteria.
Conclusion: Yes, the recording shows hypertrophy.
<END>

Generation rules
- Write only about the recording. The Evidence, Reasoning and Conclusion must read
  as reasoning about an ECG and nothing else. Never mention the known answer, the
  label, the correct answer, what is required, the written report, the clinical
  interpretation, or the annotations, in any phrasing. Do not write "the report
  does not describe", "not captured in the written interpretation", "the known
  answer", or equivalents -- these are facts about how the supervision was
  produced, not about the ECG.
- Use only information supported by the provided input.
- Do not hallucinate diagnoses, findings, measurements, or clinical history.
- Each reasoning step must explicitly reference Evidence items.
- Do not infer or mention prior events, comorbidities, medications, or management.
  No investigations, no treatment suggestions, no follow-up advice.
- Age and sex may be referenced as given patient context. Never present them as
  inferred from the recording.
- Signal-quality annotations may be used to qualify confidence in a finding.
- Use cautious clinical language such as "suggests", "consistent with", or
  "likely" when appropriate. Avoid unsupported severity claims.
- Where a finding carries a low annotation likelihood, reflect that uncertainty in
  the reasoning rather than stating it flatly.
- Each Conclusion must begin with Yes or No followed by a comma, and must agree
  with that question's known answer. For a No answer, phrase it as "the recording
  as described shows no findings of X" rather than "does not show evidence of X".
  For the NORM question, a No instead reads "No, the recording as described is not
  a normal ECG". Keep the "Conclusion: No," prefix exactly.
- At most 5 Evidence items per block. Keep each block under 200 words.

---
{input_block}"""


def _questions_block(view: TeacherView) -> str:
    """The numbered question list with known answers for the Mode B Task section."""
    lines = []
    for i, sc in enumerate(SUPERCLASSES, start=1):
        answer = "Yes" if superclass_answer(view, sc) else "No"
        lines.append(f"  {i}. {sc} - {question_text(sc)} Known answer: {answer}")
    return "\n".join(lines)


def render_prompt_mode_b(view: TeacherView) -> str:
    """Render the single Mode B prompt covering all five superclasses."""
    return MODE_B_TEMPLATE.format(
        questions_block=_questions_block(view),
        input_block=_render_input_block(view),
    )


def render_record_mode_b(rec: dict) -> str:
    """Convenience: build a view from a manifest entry and render the Mode B prompt."""
    return render_prompt_mode_b(build_view(rec))
