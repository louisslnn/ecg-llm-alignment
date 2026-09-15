"""Async generation client for the Mode B teacher run (Phase 1.5).

Talks to an OpenAI-compatible ``/chat/completions`` endpoint (base URL and key
from the environment / a gitignored ``.env``) and drives one call per record,
each covering all five superclasses.

Design points, all required by the task:

* **Mode B only.** One request per record, prompt from
  :func:`src.teacher.prompt.render_prompt_mode_b`, five blocks in one response.
* **``max_tokens`` defaults to 3000.** gpt-oss spends a large share of its budget
  on a separate reasoning channel; a truncated visible response is unparseable, so
  the ceiling is generous.
* **Read ``choices[0].message.content`` and handle ``null`` explicitly.** A null
  or empty content (typically ``finish_reason == "length"`` with the whole budget
  spent on reasoning) is routed to the error file, not crashed on and not written
  to the output (so a rerun retries it).
* **Persist the raw content, the reasoning field, ``finish_reason`` and usage.**
  The parsed result is stored too, but never *instead* of the raw content, so a
  parser fix (:mod:`src.teacher.parse`) never forces regeneration.
* **Store the cleaned content and both parses.** ``cleaned_content`` is the raw
  content with supervision-pipeline Reasoning steps stripped
  (:func:`src.teacher.postprocess.strip_content`). ``parsed`` is parsed from the
  raw content (diagnostics only); ``parsed_clean`` is parsed from
  ``cleaned_content`` and is the canonical parse **Phase 2 target assembly must
  read from** -- never ``parsed``, which still holds the unstripped Reasoning. A
  strip invariant is asserted: ``parsed_clean`` must contain no supervision
  pipeline tell, and any survivor is counted and reported in the run summary.
* **Concurrency cap, retry on 5xx/429/timeouts with backoff, SKIP_EXISTING by
  ecg_id against the output jsonl, a separate error file, throughput logging.**

Uses ``httpx`` directly (rather than the openai SDK) so non-standard fields like
``reasoning`` survive untouched and ``content: null`` is visible as-is.
"""

import asyncio
import json
import logging
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

from . import parse as P
from . import postprocess as PP
from .prompt import PROMPT_VERSION, render_prompt_mode_b, superclass_answers
from .view import build_view

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_MAX_TOKENS = 3000
DEFAULT_CONCURRENCY = 8
MODE = "B"

# Retryable transport-level HTTP statuses (rate limit + server errors).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def load_dotenv(path: str = ".env") -> None:
    """Minimal ``.env`` loader (no python-dotenv dependency).

    Populates os.environ for KEY=VALUE lines it does not already contain. Missing
    file is a no-op. Values may be single/double quoted; ``export`` prefixes and
    ``#`` comments are handled.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


@dataclass
class TeacherConfig:
    base_url: str
    api_key: str
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 1.0
    top_p: float = 1.0
    reasoning_effort: Optional[str] = None  # sent only if set (e.g. "low"/"high")
    concurrency: int = DEFAULT_CONCURRENCY
    request_timeout: float = 180.0
    max_retries: int = 5
    backoff_base: float = 2.0
    backoff_cap: float = 60.0

    @property
    def endpoint(self) -> str:
        """The chat-completions URL.

        Tolerates a base URL given either as the API root (``.../v1``) or as the
        full endpoint (``.../v1/chat/completions``) so both spellings in .env work.
        """
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    @classmethod
    def from_env(cls, load_env_file: bool = True, **overrides: Any) -> "TeacherConfig":
        """Build config from environment. Reads ``.env`` first if present.

        Base URL: ``TEACHER_BASE_URL`` or ``OPENAI_BASE_URL``.
        API key:  ``TEACHER_API_KEY`` or ``OPENAI_API_KEY``.
        """
        if load_env_file:
            load_dotenv()
        base_url = os.environ.get("TEACHER_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("TEACHER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not base_url:
            raise RuntimeError(
                "no base URL: set TEACHER_BASE_URL or OPENAI_BASE_URL "
                "(e.g. in a .env file)"
            )
        if not api_key:
            raise RuntimeError(
                "no API key: set TEACHER_API_KEY or OPENAI_API_KEY (e.g. in a .env file)"
            )
        model = os.environ.get("TEACHER_MODEL", DEFAULT_MODEL)
        return cls(base_url=base_url, api_key=api_key, model=model, **overrides)


def read_existing_ids(path: str) -> set:
    """ecg_ids already present in the output jsonl (for SKIP_EXISTING)."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["ecg_id"]))
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
    return done


class _JsonlWriter:
    """Append-only jsonl writer guarded by an async lock, flushed per line."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self._lock = asyncio.Lock()

    async def write(self, obj: Dict[str, Any]) -> None:
        async with self._lock:
            self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _build_payload(config: TeacherConfig, prompt: str) -> Dict[str, Any]:
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    if config.reasoning_effort:
        payload["reasoning_effort"] = config.reasoning_effort
    return payload


def _extract_reasoning(message: Dict[str, Any]) -> Optional[str]:
    """Reasoning channel, however this endpoint spells it."""
    for key in ("reasoning", "reasoning_content"):
        val = message.get(key)
        if val:
            return val
    return None


async def _post_with_retry(
    client: httpx.AsyncClient, config: TeacherConfig, payload: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], int]:
    """POST with backoff. Returns (json_data, error_info, attempts).

    Retries timeouts, transport errors and 429/5xx up to ``max_retries``.
    Non-retryable 4xx and exhausted retries return an error_info dict.
    """
    headers = {"Authorization": f"Bearer {config.api_key}"}
    last_error: Dict[str, Any] = {}
    for attempt in range(1, config.max_retries + 1):
        try:
            resp = await client.post(
                config.endpoint, json=payload, headers=headers,
                timeout=config.request_timeout,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = {"error": "timeout" if isinstance(exc, httpx.TimeoutException)
                          else "transport", "status_code": None, "message": str(exc)}
        else:
            if resp.status_code < 300:
                return resp.json(), None, attempt
            body = resp.text[:500]
            last_error = {
                "error": f"http_{resp.status_code}",
                "status_code": resp.status_code,
                "message": body,
            }
            if resp.status_code not in _RETRYABLE_STATUS:
                return None, {**last_error, "attempts": attempt}, attempt  # non-retryable

        if attempt < config.max_retries:
            delay = min(config.backoff_cap, config.backoff_base ** attempt)
            delay *= 0.5 + random.random()  # jitter
            logger.warning(
                "attempt %d/%d failed (%s); retrying in %.1fs",
                attempt, config.max_retries, last_error.get("error"), delay,
            )
            await asyncio.sleep(delay)

    return None, {**last_error, "attempts": config.max_retries}, config.max_retries


async def _generate_record(
    client: httpx.AsyncClient,
    config: TeacherConfig,
    rec: Dict[str, Any],
    sem: asyncio.Semaphore,
    out_writer: _JsonlWriter,
    err_writer: _JsonlWriter,
    counters: Dict[str, int],
    leak_patterns: Counter,
    written_ids: List[int],
) -> None:
    ecg_id = int(rec["ecg_id"])
    view = build_view(rec)
    known = superclass_answers(view)
    prompt = render_prompt_mode_b(view)
    payload = _build_payload(config, prompt)

    async with sem:
        data, error, attempts = await _post_with_retry(client, config, payload)

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if error is not None:
        counters["error"] += 1
        await err_writer.write({"ecg_id": ecg_id, "created_at": now, **error})
        return

    try:
        choice = data["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content")  # may be null
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as exc:
        counters["error"] += 1
        await err_writer.write({
            "ecg_id": ecg_id, "created_at": now, "error": "malformed_response",
            "status_code": None, "message": f"{type(exc).__name__}: {exc}",
            "attempts": attempts,
        })
        return

    reasoning = _extract_reasoning(message)
    usage = data.get("usage")

    # content: null / empty -> error file so a rerun retries it (do not write output).
    if content is None or not str(content).strip():
        counters["empty"] += 1
        await err_writer.write({
            "ecg_id": ecg_id, "created_at": now, "error": "empty_content",
            "status_code": None, "finish_reason": finish_reason,
            "usage": usage, "attempts": attempts,
            "message": "content was null/empty (likely truncated on reasoning)",
        })
        return

    parsed = P.parse_response(content)
    mismatches = P.label_mismatches(parsed, known)
    leaked = P.leaked_superclasses(parsed)
    counters["ok"] += 1
    if not parsed.ok:
        counters["parse_fail"] += 1
    if mismatches:
        counters["mismatch"] += 1
        counters["mismatch_blocks"] += len(mismatches)
    if leaked:
        counters["leak_records"] += 1
        counters["leak_blocks"] += len(leaked)
        leak_patterns.update(P.leak_pattern_counts(parsed))

    # v6 fix #1: strip supervision-pipeline steps from Reasoning before caching.
    # The raw content is kept alongside so the strip is reversible.
    cleaned_content, strip_stats = PP.strip_content(content)
    counters["strip_blocks"] += strip_stats.blocks_stripped
    counters["strip_steps"] += strip_stats.steps_removed
    counters["strip_emptied"] += strip_stats.blocks_emptied

    # Parse the cleaned content too: `parsed` (from raw) is for diagnostics, while
    # `parsed_clean` (from cleaned_content) is the canonical parse Phase 2 target
    # assembly must read from -- never the raw `parsed`, which still holds the
    # unstripped Reasoning. Assert the strip left no supervision-pipeline tell in
    # parsed_clean; any survivor is counted and logged (the invariant is zero).
    parsed_clean = P.parse_response(cleaned_content)
    clean_residual = {
        sc: hits
        for sc, b in parsed_clean.blocks.items()
        if (hits := PP.find_leak_steps(b.reasoning))
    }
    if clean_residual:
        counters["clean_leak_blocks"] += len(clean_residual)
        counters["clean_leak_steps"] += sum(len(h) for h in clean_residual.values())
        logger.error(
            "supervision-pipeline text survived strip for ecg_id %d: %s",
            ecg_id, clean_residual,
        )

    # v6 output-quality checks, measured on the model's generation (raw parse).
    lead_type_blocks = sum(
        1 for b in parsed.blocks.values() if PP.has_lead_type(b.raw_block)
    )
    diagnosis_evidence_items = sum(
        PP.count_diagnostic_evidence_items(b.evidence) for b in parsed.blocks.values()
    )
    counters["lead_type_blocks"] += lead_type_blocks
    counters["diagnosis_evidence_items"] += diagnosis_evidence_items

    await out_writer.write({
        "ecg_id": ecg_id,
        "model": config.model,
        "prompt_version": PROMPT_VERSION,
        "mode": MODE,
        "created_at": now,
        "attempts": attempts,
        "known_answers": known,
        "prompt": prompt,
        "content": content,
        "cleaned_content": cleaned_content,
        "reasoning": reasoning,
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "usage": usage,
        "parsed": parsed.to_dict(),
        "parsed_clean": parsed_clean.to_dict(),
        "label_mismatches": mismatches,
        "evidence_leaks": leaked,
        "clean_supervision_residual": clean_residual,
        "strip": {
            "blocks_stripped": strip_stats.blocks_stripped,
            "steps_removed": strip_stats.steps_removed,
            "blocks_emptied": strip_stats.blocks_emptied,
        },
        "lead_type_blocks": lead_type_blocks,
        "diagnosis_evidence_items": diagnosis_evidence_items,
    })
    written_ids.append(ecg_id)


async def run_generation(
    records: List[Dict[str, Any]],
    config: TeacherConfig,
    out_path: str,
    err_path: str,
    skip_existing: bool = True,
    progress_every: int = 10,
) -> Tuple[Dict[str, int], List[int]]:
    """Generate for a list of manifest records.

    Returns ``(counters, written_ids)`` where ``written_ids`` are the ecg_ids
    successfully written to the output *in this invocation* (in completion order),
    so callers can restrict a --show-raw view to this run rather than to whatever
    a prior run left in the file.
    """
    done_ids = read_existing_ids(out_path) if skip_existing else set()
    todo = [r for r in records if int(r["ecg_id"]) not in done_ids]
    skipped = len(records) - len(todo)
    logger.info(
        "records=%d skip_existing=%s already_done=%d todo=%d concurrency=%d",
        len(records), skip_existing, skipped, len(todo), config.concurrency,
    )
    counters = {"ok": 0, "error": 0, "empty": 0, "parse_fail": 0,
                "mismatch": 0, "mismatch_blocks": 0,
                "leak_records": 0, "leak_blocks": 0,
                "strip_blocks": 0, "strip_steps": 0, "strip_emptied": 0,
                "clean_leak_blocks": 0, "clean_leak_steps": 0,
                "lead_type_blocks": 0, "diagnosis_evidence_items": 0,
                "skipped": skipped}
    leak_patterns: Counter = Counter()
    written_ids: List[int] = []
    if not todo:
        return counters, written_ids

    out_writer = _JsonlWriter(out_path)
    err_writer = _JsonlWriter(err_path)
    sem = asyncio.Semaphore(config.concurrency)
    start = time.monotonic()

    async def _wrapped(rec):
        await _generate_record(client, config, rec, sem, out_writer, err_writer,
                               counters, leak_patterns, written_ids)
        completed = counters["ok"] + counters["error"] + counters["empty"]
        if completed % progress_every == 0 or completed == len(todo):
            elapsed = time.monotonic() - start
            rate = completed / elapsed if elapsed else 0.0
            remaining = len(todo) - completed
            eta = remaining / rate if rate else float("inf")
            logger.info(
                "progress %d/%d  ok=%d empty=%d err=%d  %.2f rec/s  eta %.0fs",
                completed, len(todo), counters["ok"], counters["empty"],
                counters["error"], rate, eta,
            )

    try:
        async with httpx.AsyncClient() as client:
            await asyncio.gather(*(_wrapped(rec) for rec in todo))
    finally:
        out_writer.close()
        err_writer.close()

    elapsed = time.monotonic() - start
    logger.info(
        "done: ok=%d empty=%d error=%d parse_fail=%d mismatch=%d (%d blocks) "
        "leak_records=%d leak_blocks=%d in %.1fs (%.2f rec/s)",
        counters["ok"], counters["empty"], counters["error"], counters["parse_fail"],
        counters["mismatch"], counters["mismatch_blocks"],
        counters["leak_records"], counters["leak_blocks"], elapsed,
        (counters["ok"] + counters["empty"] + counters["error"]) / elapsed if elapsed else 0.0,
    )
    logger.info(
        "post-processing: stripped %d step(s) from %d block(s), %d left empty; "
        "quality: lead_type_blocks=%d diagnosis_evidence_items=%d",
        counters["strip_steps"], counters["strip_blocks"], counters["strip_emptied"],
        counters["lead_type_blocks"], counters["diagnosis_evidence_items"],
    )
    # Strip invariant: parsed_clean must carry no supervision-pipeline tell.
    if counters["clean_leak_blocks"]:
        logger.error(
            "STRIP INVARIANT VIOLATED: %d supervision-pipeline step(s) survived in "
            "%d parsed_clean block(s) -- Phase 2 targets would leak; inspect the run",
            counters["clean_leak_steps"], counters["clean_leak_blocks"],
        )
    else:
        logger.info("strip invariant OK: parsed_clean has no supervision-pipeline text")
    if leak_patterns:
        breakdown = ", ".join(f"{p}={n}" for p, n in leak_patterns.most_common())
        logger.warning("evidence leakage by pattern: %s", breakdown)
    return counters, written_ids
