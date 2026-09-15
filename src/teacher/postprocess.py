"""Deterministic post-processing of teacher responses (v6 fix #1).

12-13 of the pilot's 250 blocks contained a Reasoning step that told the reader the
answer had been supplied ("...yet the required answer is affirmative"). v5 banned
"known answer", so the model reworded to "required answer"; adding banned phrases
to the prompt is a losing game. Instead we delete the offending step in code,
after generation and before caching. This cannot be reworded around.

The strip operates on the Reasoning section only, step by step:

* Drop any ``Step N:`` whose text matches one of :data:`LEAK_STEP_PATTERNS`
  (case-insensitive). Every other step -- including the honest "the listed
  observations do not address hypertrophy criteria" step that usually precedes
  the tell -- is kept.
* Renumber the survivors so ``Step 1:``, ``Step 2:`` stay contiguous.
* If stripping would empty the Reasoning section, keep the block but flag it
  (:attr:`StripResult.emptied`); the caller reports the count. An empty Reasoning
  section is never emitted silently.

The raw content is always retained alongside the cleaned content (see
:func:`strip_content`), so the strip is fully reversible and can be re-run.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Supervision-pipeline tells to delete, exactly as specified in teacher_v6_fixes.md.
# These are matched case-insensitively against each Reasoning step's text.
LEAK_STEP_PATTERNS: Tuple[str, ...] = (
    r"required answer",
    r"known answer",
    r"correct answer",
    r"answer is affirmative",
    r"answer is negative",
    r"requires? an? (?:positive|negative|affirmative)",
    r"must be (?:yes|no|affirmative|negative)",
    r"the label",
    r"as (?:given|specified|stated) (?:in the )?(?:label|answer)",
    # Added after pilot review: one block ("the answer must reflect the known
    # diagnosis") reworded around the nine patterns above via "must reflect" and
    # "known diagnosis"; both are the same supervision tell.
    r"known diagnosis",
    r"answer must reflect",
)

_LEAK_STEP_RE = re.compile("|".join(LEAK_STEP_PATTERNS), re.IGNORECASE)

# A step header: optional markdown/quote decoration, "Step", the number, then an
# optional separator. Used to slice a Reasoning section into steps.
_STEP_HEADER_RE = re.compile(r"(?im)^[ \t>*#\-]*step\s*\d+\s*[:.\-—]?")
# The "Step <n>" token inside one step, for renumbering (casing/separator kept).
_STEP_NUM_RE = re.compile(r"(?i)(step\s*)\d+")


@dataclass
class StripResult:
    """Outcome of stripping one Reasoning section.

    ``cleaned`` is the renumbered Reasoning text with leaked steps removed;
    ``removed``/``kept`` count steps; ``emptied`` is True when every step was a
    tell and the section is now empty of steps (the block is kept but flagged).
    """

    cleaned: str
    removed: int
    kept: int

    @property
    def emptied(self) -> bool:
        return self.removed > 0 and self.kept == 0

    @property
    def changed(self) -> bool:
        return self.removed > 0


def is_leak_step(step_text: str) -> bool:
    """True if a single Reasoning step is a supervision-pipeline tell."""
    return bool(_LEAK_STEP_RE.search(step_text))


def _split_steps(reasoning: str) -> Tuple[str, List[str]]:
    """Split a Reasoning section into (preamble, steps).

    ``preamble`` is any text before the first ``Step`` header (usually just the
    newline that follows the ``Reasoning:`` label). Each step slice runs from its
    header up to the next header, so joining the slices reproduces the input.
    """
    matches = list(_STEP_HEADER_RE.finditer(reasoning))
    if not matches:
        return reasoning, []
    preamble = reasoning[: matches[0].start()]
    steps: List[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(reasoning)
        steps.append(reasoning[m.start():end])
    return preamble, steps


def strip_reasoning_steps(reasoning: Optional[str]) -> StripResult:
    """Delete supervision-pipeline steps from one Reasoning section and renumber.

    A section with no parseable ``Step`` headers is returned unchanged (nothing to
    strip). Leaked steps are dropped; survivors are renumbered ``Step 1``,
    ``Step 2`` ... in order, preserving each step's original casing, separator and
    body text.
    """
    if not reasoning:
        return StripResult(cleaned=reasoning or "", removed=0, kept=0)

    preamble, steps = _split_steps(reasoning)
    if not steps:
        return StripResult(cleaned=reasoning, removed=0, kept=0)

    kept_steps: List[str] = []
    removed = 0
    for step in steps:
        if is_leak_step(step):
            removed += 1
        else:
            kept_steps.append(step)

    renumbered = [
        _STEP_NUM_RE.sub(lambda m, n=i: f"{m.group(1)}{n}", step, count=1)
        for i, step in enumerate(kept_steps, start=1)
    ]
    cleaned = preamble + "".join(renumbered)
    return StripResult(cleaned=cleaned, removed=removed, kept=len(kept_steps))


# Each block's Reasoning section within a full Mode B response: the "Reasoning"
# label and inline separator (group 1), then the body up to the Conclusion, an
# <END> marker, or end of text.
_REASONING_SECTION_RE = re.compile(
    r"(?is)(\breasoning\b[ \t:*\-—]*)(.*?)(?=\bconclusion\b|<\s*END\s*>|\Z)"
)


@dataclass
class ContentStripStats:
    """Aggregate strip counts over one full multi-block response."""

    blocks_stripped: int = 0   # blocks that lost at least one step
    steps_removed: int = 0     # total steps deleted
    blocks_emptied: int = 0    # blocks whose Reasoning is now empty of steps


def strip_content(content: Optional[str]) -> Tuple[str, ContentStripStats]:
    """Strip every Reasoning section in a full Mode B response.

    Returns ``(cleaned_content, stats)``. Only Reasoning sections are touched;
    Evidence and Conclusion text is left byte-for-byte identical.
    """
    stats = ContentStripStats()
    if not content:
        return content or "", stats

    def _sub(m: "re.Match[str]") -> str:
        label, body = m.group(1), m.group(2)
        res = strip_reasoning_steps(body)
        if res.changed:
            stats.blocks_stripped += 1
            stats.steps_removed += res.removed
            if res.emptied:
                stats.blocks_emptied += 1
        return label + res.cleaned

    cleaned = _REASONING_SECTION_RE.sub(_sub, content)
    return cleaned, stats


def find_leak_steps(reasoning: Optional[str]) -> List[str]:
    """Reasoning steps that still match a supervision-pipeline tell.

    Used to assert the strip did its job: after :func:`strip_content`, re-parsing
    the cleaned text and calling this on each block's Reasoning must return an
    empty list. A non-empty result is a strip/parse invariant violation -- a tell
    the student would otherwise learn -- and is surfaced in the run summary.
    """
    _, steps = _split_steps(reasoning or "")
    return [step.strip() for step in steps if is_leak_step(step)]


# --- v6 output-quality checks (reported in the run summary) --------------------
# These are heuristic monitors, not gates: they count how often two known failure
# modes appear in a generation so the v6 run can be compared against the v5 pilot.

# Fix #2: "Lagetyp" mistranslated as "lead type". After the glossary this should
# be 0. Scanned across the whole block (Evidence, Reasoning and Conclusion).
_LEAD_TYPE_RE = re.compile(r"lead type", re.IGNORECASE)

# Fix #3: a diagnostic conclusion placed in an Evidence item ("Inferior infarct
# noted"). These markers name a diagnosis rather than an observation; deliberately
# narrow so genuine observations the prompt allows as Evidence -- bundle branch
# block, ST elevation/depression, low voltage -- are not counted.
DIAGNOSTIC_EVIDENCE_PATTERNS: Tuple[str, ...] = (
    r"infarct",
    r"infarkt",
    r"ischaem",
    r"ischem",
    r"hypertroph",
    r"hypertrofi",
    r"myocardial",
    r"myokard",
    r"ventricular strain",
    r"\bstrain\b",
    r"consistent with",
    r"atrial enlargement",
    r"chamber enlargement",
    r"p mitrale",
    r"p pulmonale",
)
_DIAGNOSTIC_EVIDENCE_RE = re.compile("|".join(DIAGNOSTIC_EVIDENCE_PATTERNS), re.IGNORECASE)

# An Evidence item boundary: a "1)" / "2." style number at the start or after
# whitespace. Splitting on it yields the individual Evidence items.
_EVIDENCE_ITEM_RE = re.compile(r"(?:^|\s)\d+\s*[\).]")


def has_lead_type(text: Optional[str]) -> bool:
    """True if a piece of text contains the "lead type" mistranslation."""
    return bool(text) and bool(_LEAD_TYPE_RE.search(text))


def split_evidence_items(evidence: Optional[str]) -> List[str]:
    """Split an Evidence section into its individual numbered items."""
    if not evidence:
        return []
    return [item.strip() for item in _EVIDENCE_ITEM_RE.split(evidence) if item.strip()]


def is_diagnostic_evidence(item: str) -> bool:
    """True if an Evidence item states a diagnostic conclusion, not an observation."""
    return bool(_DIAGNOSTIC_EVIDENCE_RE.search(item))


def count_diagnostic_evidence_items(evidence: Optional[str]) -> int:
    """Number of Evidence items in one section that state a diagnostic conclusion."""
    return sum(1 for item in split_evidence_items(evidence) if is_diagnostic_evidence(item))
