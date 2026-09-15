"""Parse a Mode B teacher response into five Evidence/Reasoning/Conclusion blocks.

The teacher is asked to emit five superclass-headed blocks (NORM, MI, STTC, CD,
HYP), each terminated by ``<END>`` (see :data:`src.teacher.prompt.MODE_B_TEMPLATE`).
This module splits the response and extracts, per superclass, the Evidence text,
the Reasoning text, and the Conclusion's Yes/No.

Design constraints from the task:
* **Report parse failures, never silently drop them.** A missing block, a block
  with no parseable Yes/No, or a response that yields no blocks at all is recorded
  in :class:`ParseResult` so the run script can count and surface it.
* **Verify the parsed Yes/No against the known label.** :func:`label_mismatches`
  returns the superclasses whose extracted conclusion disagrees with the known
  answer, so mismatches can be counted rather than trusted blindly.

Parsing is a pure function of the raw string. The raw content is persisted
separately, so this parser can be fixed and re-run without regenerating.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# A header line naming a superclass: optional markdown/quote decoration, the code,
# then end-of-line or a non-word separator (so "Conclusion: No, normal" and prose
# containing "MI" mid-sentence are not mistaken for headers).
_HEADER_RE = re.compile(
    r"(?im)^[ \t>*#\-]*(" + "|".join(SUPERCLASSES) + r")\b[ \t:*—.\-]*$"
)
_END_RE = re.compile(r"<\s*END\s*>", re.IGNORECASE)

# Conclusion line: "Conclusion: Yes, ..." / "Conclusion - No ...". The comma the
# prompt asks for is not required, so a bare "Conclusion: Yes" still parses.
_CONCLUSION_RE = re.compile(r"(?im)^[ \t>*#\-]*conclusion\b[ \t:*\-—]*(yes|no)\b")
_EVIDENCE_RE = re.compile(r"(?is)\bevidence\b[ \t:*\-—]*(.*?)(?=\breasoning\b|\bconclusion\b|\Z)")
_REASONING_RE = re.compile(r"(?is)\breasoning\b[ \t:*\-—]*(.*?)(?=\bconclusion\b|\Z)")

# Leakage patterns (case-insensitive substrings). The output must read as reasoning
# about a recording only; these catch references to the annotation channel or the
# supervision pipeline that the student never sees. Scanned across Evidence,
# Reasoning and Conclusion (see :func:`_parse_block`).
#   v2/v3: citing annotated findings or their absence.
#   v4: citing the recording-property annotation fields dropped as Evidence sources.
#   v5: supervision-pipeline / meta language (the known answer, the report, the
#       written interpretation) -- facts about how the target was produced, not the ECG.
LEAK_PATTERNS = (
    # annotation channel
    "annotat",
    "likelihood",
    "cardiologist",
    "diagnostic finding",
    "diagnostic statement",
    "not annotated",
    "heart axis",
    "infarction stadium",
    "form statement",
    # v5: supervision-pipeline / meta language
    "known answer",
    "correct answer",
    "the report",
    "written interpretation",
    "is reported",
    "not described",
)


@dataclass
class Block:
    """One parsed superclass block. Fields are None when not found."""

    superclass: str
    evidence: Optional[str]
    reasoning: Optional[str]
    conclusion: Optional[bool]  # True=Yes, False=No, None=unparseable
    raw_block: str
    # Leakage hits across Evidence, Reasoning and Conclusion:
    # [{"section", "pattern", "line"}].
    leaks: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParseResult:
    blocks: Dict[str, Block] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)          # no block found
    unparseable_conclusion: List[str] = field(default_factory=list)
    split_method: str = "none"                                # header | end | none
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when all five blocks were found with a parseable conclusion."""
        return not self.missing and not self.unparseable_conclusion and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocks": {sc: b.to_dict() for sc, b in self.blocks.items()},
            "missing": self.missing,
            "unparseable_conclusion": self.unparseable_conclusion,
            "split_method": self.split_method,
            "errors": self.errors,
            "ok": self.ok,
        }


def _clean(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = _END_RE.sub("", text).strip()
    return text or None


def _extract_conclusion(block_text: str) -> Optional[bool]:
    m = _CONCLUSION_RE.search(block_text)
    if not m:
        return None
    return m.group(1).lower() == "yes"


def detect_evidence_leakage(text: Optional[str]) -> List[Dict[str, str]]:
    """Leakage-pattern hits in a section of text.

    Returns one ``{"pattern", "line"}`` entry per (pattern, offending line) match,
    scanning line by line so the offending text is reported, not just the pattern.
    A line may match several patterns (e.g. "Infarction stadium: not annotated"
    hits ``annotat``, ``not annotated`` and ``infarction stadium``); each is
    reported.
    """
    if not text:
        return []
    hits: List[Dict[str, str]] = []
    for line in text.splitlines():
        low = line.lower()
        for pattern in LEAK_PATTERNS:
            if pattern in low:
                hits.append({"pattern": pattern, "line": line.strip()})
    return hits


def _conclusion_text(block_text: str) -> Optional[str]:
    """The Conclusion prose (everything after the Conclusion marker), if present."""
    m = re.search(r"(?is)\bconclusion\b[ \t:*\-—]*(.*)\Z", block_text)
    return _clean(m.group(1)) if m else None


def _parse_block(superclass: str, block_text: str) -> Block:
    ev = _EVIDENCE_RE.search(block_text)
    rs = _REASONING_RE.search(block_text)
    evidence = _clean(ev.group(1)) if ev else None
    reasoning = _clean(rs.group(1)) if rs else None
    conclusion_text = _conclusion_text(block_text)

    # v5: the output must read as reasoning about a recording; scan every section,
    # not just Evidence, for annotation-channel and supervision-pipeline language.
    leaks: List[Dict[str, str]] = []
    for section, section_text in (
        ("evidence", evidence),
        ("reasoning", reasoning),
        ("conclusion", conclusion_text),
    ):
        for hit in detect_evidence_leakage(section_text):
            leaks.append({"section": section, **hit})

    return Block(
        superclass=superclass,
        evidence=evidence,
        reasoning=reasoning,
        conclusion=_extract_conclusion(block_text),
        raw_block=block_text.strip(),
        leaks=leaks,
    )


def _split_by_headers(content: str) -> Optional[Dict[str, str]]:
    """Slice the response at superclass header lines. None if <2 headers found."""
    matches = list(_HEADER_RE.finditer(content))
    if len(matches) < 2:
        return None
    blocks: Dict[str, str] = {}
    for i, m in enumerate(matches):
        sc = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        # first header wins if a code is (wrongly) repeated
        blocks.setdefault(sc, content[start:end])
    return blocks


def _split_by_end(content: str) -> Optional[Dict[str, str]]:
    """Fallback: split on <END> and assign segments to superclasses in order."""
    segments = [s for s in _END_RE.split(content) if s.strip()]
    if len(segments) < 2:
        return None
    return {sc: seg for sc, seg in zip(SUPERCLASSES, segments)}


def parse_response(content: Optional[str]) -> ParseResult:
    """Parse a raw Mode B response into a :class:`ParseResult`."""
    result = ParseResult()
    if not content or not content.strip():
        result.errors.append("empty content")
        result.missing = list(SUPERCLASSES)
        return result

    raw_blocks = _split_by_headers(content)
    if raw_blocks is not None:
        result.split_method = "header"
    else:
        raw_blocks = _split_by_end(content)
        if raw_blocks is not None:
            result.split_method = "end"
        else:
            result.errors.append("no superclass headers or <END> markers found")
            result.missing = list(SUPERCLASSES)
            return result

    for sc in SUPERCLASSES:
        if sc not in raw_blocks:
            result.missing.append(sc)
            continue
        block = _parse_block(sc, raw_blocks[sc])
        result.blocks[sc] = block
        if block.conclusion is None:
            result.unparseable_conclusion.append(sc)

    return result


def leaked_superclasses(result: ParseResult) -> List[str]:
    """Superclasses whose Evidence, Reasoning or Conclusion contains any leakage."""
    return [sc for sc, b in result.blocks.items() if b.leaks]


def leak_pattern_counts(result: ParseResult) -> Dict[str, int]:
    """Per-pattern leakage hit counts across all blocks in one result."""
    counts: Dict[str, int] = {}
    for block in result.blocks.values():
        for hit in block.leaks:
            counts[hit["pattern"]] = counts.get(hit["pattern"], 0) + 1
    return counts


def label_mismatches(result: ParseResult, known: Dict[str, bool]) -> List[str]:
    """Superclasses whose parsed Yes/No disagrees with the known label.

    Blocks that are missing or whose conclusion was unparseable are not counted
    here (they are already reported as parse failures); only a confidently parsed
    conclusion that contradicts the known answer is a mismatch.
    """
    out: List[str] = []
    for sc, answer in known.items():
        block = result.blocks.get(sc)
        if block is None or block.conclusion is None:
            continue
        if block.conclusion != answer:
            out.append(sc)
    return out
