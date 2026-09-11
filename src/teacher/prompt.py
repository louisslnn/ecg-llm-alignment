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

# v2: NORM reads "Is this a normal ECG?" rather than "Does this ECG show evidence
# of a normal ECG?"; the other four keep the "show evidence of X" form.
PROMPT_VERSION = "teacher-v2"

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
- Evidence items are observations of the recording, drawn from the clinical report
  and the recording properties. Rhythm, rate, axis, intervals, waveform
  morphology, voltages, signal quality.
- The report may state a diagnosis. A diagnosis is a conclusion, not evidence. Put
  observational content in Evidence and diagnostic content in Reasoning or
  Conclusion.
- Do not invent observations. If the report describes only a rhythm, Evidence
  contains only that rhythm. Thin evidence is correct; fabricated evidence is not.
- Do not state numeric measurements (rate, intervals, amplitudes) unless they
  appear in the input.

Output format
The generated answer must follow the structure:

Evidence: 1) ... 2) ...
Reasoning:
Step 1: Using Evidence ...
Step 2: Using Evidence ...
Conclusion: Yes/No, ...
<END>

Generation rules
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
- The Conclusion must agree with the known label. Yes/No first, then a short
  explanation.
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
