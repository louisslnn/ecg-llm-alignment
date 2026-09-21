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
from collections import Counter
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


# --- report-referencing rewrite (v6 fix) --------------------------------------
# The student never sees the free-text report, so reasoning that references it
# ("the report does not describe ST elevation", "no ST changes are reported") is a
# supervision tell about a source the student cannot read. Unlike the answer-tell
# steps above, the surrounding reasoning is usually honest and worth keeping, so we
# REWRITE the offending clause in place -- reframing it onto "the listed
# observations" -- rather than deleting the whole step.
#
# The four canonical shapes and their real variants, mined from
# data/teacher/full_v6.jsonl:
#   "the report does not describe X"        -> "the listed observations do not include X"
#   "X is not described"                    -> "X is not among the listed observations"
#   "no X is reported"                      -> "the listed observations do not include X"
#   "the report provides/contains no X"     -> "the listed observations do not include X"
# plus "there is no mention of X", "... contain(s)/provide(s) no mention of X",
# "with no mention of X", and the passive families over the source verbs below.
#
# Positive references ("the report explicitly states 'normal ECG'") are NOT
# rewritten here -- reframing them grammatically is unsafe -- and are counted as
# residual by :func:`count_report_reference_residual` so nothing is hidden.

# A hyphen in any of the Unicode forms the annotators used (incl. U+2011 no-break).
_HY = "[-‐‑‒–—―−]"
# Subject: "the report", optionally joined with the signal-quality annotations.
_REPORT_SUBJECT = (
    rf"the\s+reports?"
    rf"(?:\s+(?:and|or)\s+signal{_HY}?quality\s+annotations?)?"
)
# Optional adverbs between "the report" and a positive verb ("the report also states").
_RADV = r"(?:(?:also|explicitly|clearly|merely|only|simply|further|thus|then|additionally)\s+){0,2}"
# Optional adverbs between an auxiliary/copula and the source verb.
_ADV = (
    r"(?:(?:explicitly|clearly|specifically|directly|otherwise|separately|"
    r"independently|further|additionally|expressly|overtly|positively)\s+){0,2}"
)
# Active source verbs ("the report does not <verb>").
_SRC_VERB = (
    r"(?:describe|mention|report|document|note|record|state|specify|indicate|"
    r"show|list|contain|provide|reflect|capture|detail|discuss|address|include|"
    r"characterise|characterize|reference)"
)
# Finite third-person source verbs ("the report <verb>s no ...").
_SRC_VERB_S = (
    r"(?:contains|provides|describes|mentions|documents|notes|records|lists|"
    r"shows|includes|reflects|gives|has|details|discusses|captures|indicates|"
    r"states|specifies|characterises|characterizes|references)"
)
# Past participles for the passive families ("no X is <participle>").
_SRC_PART = (
    r"(?:reported|described|mentioned|documented|noted|recorded|specified|"
    r"indicated|shown|listed|detailed|discussed|stated|captured|referenced|"
    r"characterised|characterized)"
)
# Report-referencing nouns used as "no <refnoun> of X". Deliberately excludes
# "evidence"/"indication"/"sign"/"finding": "no evidence of hypertrophy" is a
# legitimate clinical statement, not a reference to the written report.
_REFNOUN = r"(?:mention|descriptions?|documentation|notations?|records?)"

# X, a finding phrase: runs to the next clause/sentence boundary.
_X = r"(?P<x>[^.\n;:]+?)"

# An optional existence predicate after "No <refnoun> of X ___": "is present",
# "are found", "is documented" etc. Absorbed so "No description of X is present"
# rewrites cleanly to "the listed observations do not include X".
_EXIST_TAIL = (
    r"(?:\s+(?:is|are|was|were)\s+(?:present|found|seen|given|available|noted|"
    r"reported|documented|described|mentioned|recorded|shown|listed|provided|"
    r"evident|apparent|identified|observed|visible|detectable|discernible))?"
)

_INCLUDE = "the listed observations do not include"
_AMONG = "not among the listed observations"
_AMONG_POS = "among the listed observations"

# Positive report verbs, given WITHOUT the third-person "s" (captured separately):
# "the report describes X" -> "the listed observations describe X". Only verbs whose
# plural is formed by dropping the trailing "s" are listed (so "specifies"/
# "identifies" are excluded -- those only occur negated, handled above).
_POS_VERB = (
    r"(?:describe|state|characterise|characterize|note|record|report|show|label|"
    r"indicate|mention|provide|contain|reflect|document|detail|list|reference|"
    r"discuss|address|capture)"
)
# Positive passive participles ("atrial fibrillation is reported"). Narrower than
# _SRC_PART: only participles that read cleanly as "... is among the listed
# observations" once reframed.
_POS_PART = r"(?:reported|described|documented|recorded|noted|mentioned|stated)"
# Participles that take "... as Y" (a characterisation): "reported as left-axis".
_CHAR_PART = (
    r"(?:reported|described|characterised|characterized|labelled|labeled|noted|"
    r"recorded|documented|classified)"
)
# Nouns that would double the subject if left leading the object of "do not
# include" ("... do not include observations of X"). Collapsed post-hoc (fix 2).
_COLLAPSE = (
    r"(?:observations?|mention|descriptions?|documentation|notations?|records?|"
    r"information|findings?)"
)

# Optionally absorb a trailing "in the (written) report/interpretation" so
# "no X is mentioned in the report" rewrites cleanly. "in the observations",
# "in Evidence 1", "in leads V1-V3" are NOT matched here and, thanks to the
# following negative lookahead, keep those passives untouched.
_IN_REPORT = r"(?:\s+in\s+the\s+(?:written\s+)?(?:report|interpretation))?"


def _match_case(source: str, replacement: str) -> str:
    """Uppercase the first letter of ``replacement`` iff ``source`` starts capital."""
    m = re.search(r"[A-Za-z]", source)
    if not (m and source[m.start()].isupper()):
        return replacement
    chars = list(replacement)
    for i, ch in enumerate(chars):
        if ch.isalpha():
            chars[i] = ch.upper()
            return "".join(chars)
    return replacement


def _fixed(target: str):
    """Replacement callable that emits a fixed clause, case-matched to the source."""
    return lambda m: _match_case(m.group(0), target)


def _include_x(m: "re.Match[str]") -> str:
    return _match_case(m.group(0), f"{_INCLUDE} {m.group('x').strip()}")


def _among_keep_copula(m: "re.Match[str]") -> str:
    # "<is|are> not <participle>" -> "<is|are> not among the listed observations"
    return f"{m.group('cop')} {_AMONG}"


def _aux_no_mention(m: "re.Match[str]") -> str:
    singular = {"contains", "provides", "includes", "gives", "makes"}
    verb = m.group("v").lower()
    base = "does not include" if verb in singular else "do not include"
    return _match_case(m.group(0), base)


def _recase_from(source: str, rest: str) -> str:
    """Strip ``rest``; capitalize its first letter iff ``source`` began capital."""
    rest = rest.strip()
    m = re.search(r"[A-Za-z]", source)
    if m and source[m.start()].isupper():
        mr = re.search(r"[a-z]", rest)
        if mr:
            i = mr.start()
            rest = rest[:i] + rest[i].upper() + rest[i + 1:]
    return rest


def _keep_rest(m: "re.Match[str]") -> str:
    """Drop the matched attribution, keep and re-case the captured remainder."""
    return _recase_from(m.group(0), m.group("rest"))


def _tracing_is(m: "re.Match[str]") -> str:
    # "the report describes the tracing as " -> "the tracing is " (trailing space
    # preserved so the complement that follows stays separated).
    return f"{_recase_from(m.group(0), m.group('x'))} is "


# For the positive-citation drop, only delete "the report <verb>s" when what
# follows is a clause of its own -- a subject then a finite verb -- so "the report
# states the tracing is normal" -> "the tracing is normal", but "the report mentions
# only the rhythm" (a bare noun phrase) is left untouched and counted as residual.
_CLAUSE_SUBJ = r"(?:the|this|that|it|its|these|those|a|an|both|there)"
# A finite verb (copula or common lexical). Deliberately excludes do/does/did so a
# dangling second predicate ("... and does not mention X") is not read as a clause.
_FINITE_VERB = (
    r"(?:is|are|was|were|has|have|had|shows?|reveals?|indicates?|implies|includes?|"
    r"contains?|reflects?|demonstrates?|suggests?|confirms?|denotes?|appears?|"
    r"remains?|represents?|exhibits?|displays?|means?|supports?|describes?)"
)

# The remainder of a possessive deletion starts "clean" (a determiner, pronoun,
# quote, or number) and needs no article; otherwise it opens on a bare noun and we
# insert "the" ("wording of X implies" -> "the wording of X implies").
_CLEAN_START = re.compile(
    r"^(?:[\"'“”„‘’]|\d|(?:the|a|an|this|that|these|those|its|it|they|no|any|some|"
    r"each|every|all|both|his|her|their)\b)",
    re.I,
)


def _possessive_drop(m: "re.Match[str]") -> str:
    """Drop "the report's"; if the remainder is a bare noun, prepend "the"."""
    rest = m.group("rest").strip()
    if not _CLEAN_START.match(rest):
        rest = "the " + rest
    return _recase_from(m.group(0), rest)


def _among_positive(m: "re.Match[str]") -> str:
    # "X is reported" -> "X is among the listed observations"
    return f"{m.group('cop')} {_AMONG_POS}"


# Ordered rewrite rules. Order matters: the "the report ..." and the finite
# "no mention of" prefixes are consumed before the capture-based passive and bare
# rules, so no clause is rewritten twice.
REWRITE_RULES: Tuple[Tuple[str, "re.Pattern[str]", object], ...] = (
    # 1. the report (and/or annotations) do/does/did not <verb> [<refnoun> of]  ...
    ("report_active",
     re.compile(rf"\b{_REPORT_SUBJECT}\s+(?:does|do|did)\s+not\s+{_ADV}{_SRC_VERB}"
                rf"(?:\s+(?:any|a|an)?\s*{_REFNOUN}\s+of)?\b", re.I),
     _fixed(_INCLUDE)),
    # 2. the report (and/or annotations) <verb>s no [<refnoun> of]  ...
    ("report_verb_no",
     re.compile(rf"\b{_REPORT_SUBJECT}\s+{_SRC_VERB_S}\s+no(?:\s+{_REFNOUN}\s+of)?\b", re.I),
     _fixed(_INCLUDE)),
    # 2a. POSITIVE citation, "describes ... as" shape: the report describes the
    #     tracing as Y  ->  the tracing is Y. Substituting "the listed observations"
    #     here is self-referential, so we drop the attribution and keep the claim.
    ("report_positive_as",
     re.compile(rf"\b{_REPORT_SUBJECT}\s+{_RADV}"
                rf"(?:describes|characterises|characterizes|labels|classifies|reports|depicts)\s+"
                rf"(?P<x>the\s+(?:overall\s+)?(?:tracing|ecg|ekg|recording|rhythm|study|record))"
                rf"\s+as\s+", re.I),
     _tracing_is),
    # 2b. POSITIVE citation, general: the report <verb>s REST  ->  REST, but ONLY
    #     when REST is a clause (subject + finite verb). "the report states the
    #     tracing is normal" -> "the tracing is normal"; "the report mentions only
    #     the rhythm" (a bare noun phrase) is left untouched -> residual.
    ("report_positive",
     re.compile(rf"\b{_REPORT_SUBJECT}\s+{_RADV}(?:{_POS_VERB})s\s+(?:that\s+)?"
                rf"(?P<rest>{_CLAUSE_SUBJ}\b[^.;:\n]*?\b{_FINITE_VERB}\b[^.;:\n]*)", re.I),
     _keep_rest),
    # 2c. POSITIVE possessive: the report's REST  ->  REST (attribution dropped;
    #     "the" inserted when the remainder opens on a bare noun).
    ("report_possessive",
     re.compile(r"\bthe\s+reports?['’]s\s+(?P<rest>[^.;:\n]+)", re.I),
     _possessive_drop),
    # 3. there is/are no <refnoun> of  ...
    ("there_no_mention",
     re.compile(rf"\bthere\s+(?:is|are|was|were)\s+no\s+{_REFNOUN}\s+of\b", re.I),
     _fixed(_INCLUDE)),
    # 4. <subject> contain(s)/provide(s)/... no <refnoun> of  ...
    ("aux_no_mention",
     re.compile(rf"\b(?P<v>contains|provides|includes|gives|makes|contain|provide|"
                rf"include|give|make)\s+no\s+{_REFNOUN}\s+of\b", re.I),
     _aux_no_mention),
    # 5. providing/containing/... no <refnoun> of  ->  finite clause "and does not
    #    include ..." (fix 3: "not including X" read as a claim about the ECG).
    ("participle_no_mention",
     re.compile(rf"\b(?:providing|containing|showing|giving|making|indicating|"
                rf"offering)\s+no\s+{_REFNOUN}\s+of\b", re.I),
     _fixed("and does not include")),
    # 6. with no <refnoun> of X  ->  with no X among the listed observations
    ("with_no_mention",
     re.compile(rf"\bwith\s+no\s+{_REFNOUN}\s+of\s+{_X}(?=[.\n;:]|$)", re.I),
     lambda m: f"with no {m.group('x').strip()} among the listed observations"),
    # 7. no X is/are [adv] <participle> [in the report]. The trailing "in the
    #    report" is consumed; "... in Evidence 1 / in the observations / in leads"
    #    is left alone (that already points at the observations, not a document).
    ("passive_no_x",
     re.compile(rf"\bno\s+(?P<x>[^.\n;:]+?)\s+(?:is|are|was|were)\s+{_ADV}{_SRC_PART}"
                rf"{_IN_REPORT}\b(?!\s+(?:in|as)\b)", re.I),
     _include_x),
    # 8. <X> is/are not [adv] <participle> [in the report]  ->  ... not among the
    #    listed observations
    ("passive_not_x",
     re.compile(rf"\b(?P<cop>is|are|was|were)\s+not\s+{_ADV}{_SRC_PART}"
                rf"{_IN_REPORT}\b(?!\s+(?:in|as)\b)", re.I),
     _among_keep_copula),
    # 9. leftover bare "no <refnoun> of X [is present]" (after ; , and but ...)
    ("bare_no_mention",
     re.compile(rf"\bno\s+{_REFNOUN}\s+of\s+{_X}{_EXIST_TAIL}(?=[.\n;:]|$)", re.I),
     _include_x),
    # 10. POSITIVE characterisation: X is/are <part> as Y  ->  X is/are Y.
    #     "the axis is reported as left-axis deviation" -> "the axis is left-axis
    #     deviation". "not <part> as" never matches (the "not" breaks adjacency).
    ("characterisation_as",
     re.compile(rf"\b(?P<cop>is|are|was|were)\s+{_ADV}{_CHAR_PART}\s+as\b", re.I),
     lambda m: m.group("cop")),
    # 11. POSITIVE passive: X is/are <part>  ->  X is/are among the listed
    #     observations. Runs after every negative passive; "... as/in/to/by ..." is
    #     left for rule 10 or as a genuine location.
    ("passive_positive",
     re.compile(rf"\b(?P<cop>is|are|was|were)\s+{_ADV}{_POS_PART}\b(?!\s+(?:as|in|to|by)\b)", re.I),
     _among_positive),
    # 12. POSITIVE locative: "<finding> [is] <participle> in the report" cites the
    #     report for something it does say -> drop the attribution, keep the finding
    #     ("appearance described in the report" -> "appearance"). Deleting, not
    #     substituting, avoids the self-referential "in the listed observations".
    ("in_the_report",
     re.compile(rf"(?:\s+(?:is|are|was|were))?(?:\s+{_ADV}{_SRC_PART})?"
                rf"\s+in\s+the\s+(?:written\s+)?reports?\b", re.I),
     ""),
)


# Fix 2: when a rewrite leaves "the listed observations do not include observations
# of / mention of / description of X", the leading noun doubles the subject. Collapse.
_DEDUP_RE = re.compile(
    rf"\b({re.escape(_INCLUDE)})\s+{_COLLAPSE}\s+of\b", re.I
)


def rewrite_report_references(text: Optional[str]) -> Tuple[str, Counter]:
    """Rewrite report-referencing clauses in one piece of text.

    Returns ``(rewritten, per_rule_counts)`` where ``per_rule_counts[name]`` is the
    number of substitutions rule ``name`` made. The text is otherwise untouched.
    """
    counts: Counter = Counter()
    if not text:
        return text or "", counts
    out = text
    for name, pattern, repl in REWRITE_RULES:
        out, n = pattern.subn(repl, out)
        if n:
            counts[name] += n
    out, nd = _DEDUP_RE.subn(lambda m: m.group(1), out)  # fix 2: collapse doubled nouns
    if nd:
        counts["dedup_double_noun"] += nd
    return out, counts


# Detects any surviving report-referencing phrase, for the residual monitor. This
# is broader than the rewrite rules (it also matches positive references we do not
# rewrite), so a non-zero count after rewriting is expected and reported, not fatal.
# A passive participle followed by "in the observations / in Evidence / in leads"
# points at the observations, not the report, so it is NOT residual.
_OK_LOC = r"(?!\s+in\s+(?:the\s+|any\s+|either\s+)?(?:observ|evidence|tracing|recording|lead|signal))"

# Residual report-referencing patterns, named so the re-scan can break the count
# down by pattern (fix 4). "in the report" is reported under its own key, so the
# bare "the_report" key excludes an immediately preceding "in ".
_RESIDUAL_PATTERNS = {
    "the_report": re.compile(rf"(?<!in\s)\bthe\s+reports?(?:['’]s)?\b", re.I),
    "written_report": re.compile(r"\bwritten\s+(?:report|interpretation)\b", re.I),
    "in_the_report": re.compile(r"\bin\s+the\s+(?:written\s+)?reports?\b", re.I),
    "noun_of": re.compile(rf"\bno\s+{_REFNOUN}\s+of\b", re.I),
    # "X is not <part>" (a real existence tell), but not "not <part> as Y" (a
    # characterisation we deliberately leave) and not "... in the observations".
    "passive_not": re.compile(
        rf"\b(?:is|are|was|were)\s+not\s+{_ADV}{_SRC_PART}\b(?!\s+as\b){_OK_LOC}", re.I),
    "passive_no": re.compile(
        rf"\bno\s+[^.\n;:]+?\s+(?:is|are|was|were)\s+{_ADV}{_SRC_PART}\b{_OK_LOC}", re.I),
    "passive_positive": re.compile(
        rf"\b(?:is|are|was|were)\s+{_ADV}{_POS_PART}\b(?!\s+(?:as|in|to|by)\b){_OK_LOC}", re.I),
}


def report_reference_pattern_hits(text: Optional[str]) -> dict:
    """Which residual report-referencing patterns still match, with match counts."""
    if not text:
        return {}
    return {n: len(rx.findall(text)) for n, rx in _RESIDUAL_PATTERNS.items() if rx.search(text)}


def count_report_reference_residual(text: Optional[str]) -> int:
    """How many report-referencing phrases remain (incl. positive, un-rewritten)."""
    if not text:
        return 0
    return sum(len(rx.findall(text)) for rx in _RESIDUAL_PATTERNS.values())


@dataclass
class RewriteStats:
    """Aggregate rewrite counts over one full multi-block response."""

    blocks_rewritten: int = 0                 # blocks that changed
    substitutions: int = 0                    # total clause rewrites
    per_rule: Optional[dict] = None           # rule name -> substitutions

    def __post_init__(self):
        if self.per_rule is None:
            self.per_rule = {}


def rewrite_content(content: Optional[str]) -> Tuple[str, RewriteStats]:
    """Rewrite every report-referencing clause across a full response.

    Applied to ``cleaned_content`` after step-stripping; the result is what
    :func:`src.teacher.parse.parse_response` should re-parse into ``parsed_clean``.
    """
    stats = RewriteStats()
    if not content:
        return content or "", stats
    rewritten, counts = rewrite_report_references(content)
    stats.substitutions = sum(counts.values())
    stats.per_rule = dict(counts)
    return rewritten, stats


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
