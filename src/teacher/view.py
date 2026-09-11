"""Assemble the teacher's view of a single record from a manifest entry.

Deterministic and serialisable, and deliberately free of any prompt text --
:mod:`src.teacher.prompt` renders a view into the final string. Keeping the two
apart lets us inspect and test exactly what evidence the teacher is given,
independent of how it is worded.

A view carries only what the teacher is permitted to reason from: the cleaned
report, the annotated SCP statements (description, class flags, diagnostic
superclass, likelihood), recording properties (heart axis, infarction stadium,
signal-quality annotations), and patient context (age, sex). Likelihoods are
preserved exactly, including ``0.0`` -- the prompt, not the view, encodes the
rule that a likelihood of 0 means "unstated", not "absent".
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .report import DEFAULT_MIN_REPORT_CHARS, CleanedReport, ReportFlags, clean_report

# PTB-XL encodes sex as 0/1. See the dataset documentation.
_SEX_LABELS = {0: "male", 1: "female"}

# The class-flag columns, in the order the teacher prompt lists them.
_STATEMENT_CLASSES = ("diagnostic", "form", "rhythm")

# The two infarction-stadium slots in the manifest. "unknown" carries no
# information (the annotator marked the stadium as indeterminate), so it is
# dropped alongside empty slots.
_STADIUM_FIELDS = ("infarction_stadium1", "infarction_stadium2")
_STADIUM_DROP = {"unknown"}

# --- signal-quality parsing --------------------------------------------------
# The four signal-quality columns are free-text, comma-separated lead lists as
# typed by annotators. They are messy: a leading empty value (" , V6"), trailing
# padding ("v6,  "), lead ranges ("I-AVF", "I - AVF", "v1-6"), semicolon
# separators ("V5;V6"), lowercase leads ("v6"), lead+severity in one token
# ("V1 stark"), uncertainty marks ("V1???"), German "all-leads" words ("alles",
# "alle Abl.") and non-lead descriptors ("mittel", "noisy recording"). We parse
# out the leads we can recognise, normalise them to standard casing, and render
# readable prose; everything unrecognised is dropped.

_SIGNAL_QUALITY_FIELDS = (
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
)

_QUALITY_LABELS = {
    "baseline_drift": "baseline drift",
    "static_noise": "static noise",
    "burst_noise": "burst noise",
    "electrodes_problems": "electrode problems",
}

# Limb leads, lower-cased key -> standard casing. Chest leads V1..V9 are handled
# by pattern. aVR/aVL/aVF get their conventional lower-a spelling.
_LIMB_LEADS = {"i": "I", "ii": "II", "iii": "III", "avr": "aVR", "avl": "aVL", "avf": "aVF"}

# Whole segments meaning "all leads" (German "alles"/"alle"/"alle Abl.").
_ALL_LEADS_TOKENS = {"alles", "alle", "alle abl.", "alle abl", "all"}

_SEGMENT_SPLIT_RE = re.compile(r"[;,]")       # comma or semicolon between entries
_RANGE_SPLIT_RE = re.compile(r"\s*-+\s*")     # dash range, tolerating spaces/repeats
_CHEST_LEAD_RE = re.compile(r"v\s*([1-9])$")  # "v6", "V 4"


def _normalise_single_lead(token: str) -> Optional[str]:
    """One lead token -> standard casing, or None if it is not a recognisable lead.

    Trailing uncertainty/punctuation ("V1???", "V5 stark!") is trimmed; a bare
    single digit is read as the corresponding chest lead ("6" -> "V6"), which is
    what a lone number means in these lead-list fields.
    """
    t = token.strip().strip("?!.").strip().lower()
    if not t:
        return None
    if t in _LIMB_LEADS:
        return _LIMB_LEADS[t]
    m = _CHEST_LEAD_RE.fullmatch(t)
    if m:
        return f"V{m.group(1)}"
    if re.fullmatch(r"[1-9]", t):
        return f"V{t}"
    return None


def _normalise_lead_entry(entry: str) -> Optional[str]:
    """A single entry that may be one lead or a range ("I-AVF", "v1-6")."""
    entry = entry.strip()
    if "-" not in entry:
        return _normalise_single_lead(entry)
    parts = _RANGE_SPLIT_RE.split(entry)
    if len(parts) != 2:
        return None
    left = _normalise_single_lead(parts[0])
    right = _normalise_single_lead(parts[1])
    if left and right:
        return f"{left}-{right}"
    return None


def _parse_quality_leads(raw: str) -> Tuple[List[str], bool]:
    """Return (ordered unique leads, all_leads_flag) parsed from a raw field."""
    leads: List[str] = []
    all_flag = False
    for segment in _SEGMENT_SPLIT_RE.split(raw):
        s = segment.strip()
        if not s:
            continue
        if s.lower() in _ALL_LEADS_TOKENS:
            all_flag = True
            continue
        lead = _normalise_lead_entry(s)
        if lead is not None:
            if lead not in leads:
                leads.append(lead)
            continue
        # A mixed token like "V1 stark" or "mittel I": pull out any lead word.
        for word in s.split():
            lead = _normalise_lead_entry(word)
            if lead is not None and lead not in leads:
                leads.append(lead)
    return leads, all_flag


def render_quality_field(field_name: str, raw: Any) -> Optional[str]:
    """Render one signal-quality field to prose, or None if it is not annotated.

    "baseline drift in leads II, III", "static noise in leads I-aVF", "static
    noise in all leads", or -- when the field is present but names no recoverable
    lead -- "static noise annotated (leads not specified)".
    """
    if not (isinstance(raw, str) and raw.strip()):
        return None
    label = _QUALITY_LABELS.get(field_name, field_name)
    leads, all_flag = _parse_quality_leads(raw)
    if leads:
        return f"{label} in leads {', '.join(leads)}"
    if all_flag:
        return f"{label} in all leads"
    return f"{label} annotated (leads not specified)"


@dataclass(frozen=True)
class StatementView:
    """One annotated SCP statement as the teacher sees it."""

    code: str
    description: Optional[str]
    classes: List[str]  # subset of {"diagnostic", "form", "rhythm"}
    diagnostic_superclass: Optional[str]  # NORM/MI/STTC/CD/HYP, or None
    likelihood: float


@dataclass(frozen=True)
class TeacherView:
    """Everything the teacher is given about one record, minus the wording."""

    ecg_id: int
    report_text: str
    has_usable_report: bool
    report_flags: ReportFlags
    statements: List[StatementView]
    heart_axis: Optional[str]
    infarction_stadium: List[str]
    signal_quality: List[str]  # rendered prose, one entry per annotated field
    age: Optional[float]
    sex: Optional[str]
    sex_code: Optional[int] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)  # dataclasses (StatementView, ReportFlags) recurse cleanly
        return d


def _statement_view(stmt: Dict[str, Any]) -> StatementView:
    classes = [c for c in _STATEMENT_CLASSES if stmt.get(c)]
    return StatementView(
        code=stmt["code"],
        description=stmt.get("description"),
        classes=classes,
        # Only diagnostic statements carry a superclass; guard so a stray
        # diagnostic_class on a non-diagnostic row never leaks in.
        diagnostic_superclass=(
            stmt.get("diagnostic_class") if "diagnostic" in classes else None
        ),
        likelihood=float(stmt["likelihood"]),
    )


def _infarction_stadium(rec: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for f in _STADIUM_FIELDS:
        v = rec.get(f)
        if isinstance(v, str) and v.strip() and v.strip() not in _STADIUM_DROP:
            out.append(v.strip())
    return out


def _signal_quality(rec: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for f in _SIGNAL_QUALITY_FIELDS:
        phrase = render_quality_field(f, rec.get(f))
        if phrase is not None:
            out.append(phrase)
    return out


def build_view(
    rec: Dict[str, Any], min_report_chars: int = DEFAULT_MIN_REPORT_CHARS
) -> TeacherView:
    """Build a :class:`TeacherView` from a manifest entry.

    The report is cleaned via :func:`clean_report`; when no usable report
    survives, ``report_text`` is empty and ``has_usable_report`` is ``False``,
    signalling downstream (and the prompt) to rely on the annotations alone.
    """
    cleaned: CleanedReport = clean_report(rec.get("report"), min_chars=min_report_chars)

    sex_code = rec.get("sex")
    sex_code = int(sex_code) if isinstance(sex_code, (int, float)) else None

    heart_axis = rec.get("heart_axis")
    heart_axis = heart_axis.strip() if isinstance(heart_axis, str) and heart_axis.strip() else None

    return TeacherView(
        ecg_id=int(rec["ecg_id"]),
        report_text=cleaned.text,
        has_usable_report=cleaned.has_usable_report,
        report_flags=cleaned.flags,
        statements=[_statement_view(s) for s in rec.get("statements", [])],
        heart_axis=heart_axis,
        infarction_stadium=_infarction_stadium(rec),
        signal_quality=_signal_quality(rec),
        age=rec.get("age"),
        sex=_SEX_LABELS.get(sex_code),
        sex_code=sex_code,
    )
