"""Embedding-cache index helpers: hashing, provenance, staleness detection.

The cache index (``index.json``) carries a header recording the manifest hash and
the git commit the cache was built from, so a cache built against a different
manifest or different encoder code is detectable rather than silently resumed.
"""

import hashlib
import json
import os
import subprocess
from typing import Any, Dict, Optional

INDEX_NAME = "index.json"


class StaleCacheError(RuntimeError):
    """Raised when an existing cache's manifest hash no longer matches."""

    def __init__(self, stored: Optional[str], current: str):
        self.stored = stored
        self.current = current
        super().__init__(
            "cache is stale: index.json was built from manifest "
            f"{stored} but the current manifest hashes to {current}"
        )


def hash_file(path: str, algo: str = "sha256", chunk: int = 1 << 20) -> str:
    """Streamed hex digest of a file (does not load it all into memory)."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit_if_clean(repo_root: str) -> Optional[str]:
    """HEAD commit sha iff the working tree is clean, else None.

    A dirty tree (or no git) returns None so we never record a commit that does
    not actually describe the code that produced the cache.
    """
    def _git(*args):
        return subprocess.run(
            ["git", "-C", repo_root, *args],
            capture_output=True, text=True, check=True,
        ).stdout

    try:
        if _git("status", "--porcelain").strip():
            return None  # uncommitted changes -> not traceable
        return _git("rev-parse", "HEAD").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def index_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, INDEX_NAME)


def load_index(cache_dir: str) -> Dict[str, Any]:
    with open(index_path(cache_dir), encoding="utf-8") as f:
        return json.load(f)


def write_index(cache_dir: str, index: Dict[str, Any]) -> None:
    """Atomically write index.json (tmp then rename)."""
    p = index_path(cache_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f)
    os.replace(tmp, p)


def check_resume(cache_dir: str, manifest_path: str, force: bool = False
                 ) -> Optional[Dict[str, Any]]:
    """Decide whether resuming from an existing cache index is allowed.

    Returns None when there is no existing index (a fresh build). Otherwise it
    compares the index's stored manifest hash to the current manifest file:

    * hashes match -> returns the index (safe to resume);
    * hashes differ and ``force`` is False -> raises :class:`StaleCacheError`;
    * hashes differ and ``force`` is True -> returns the index (override).
    """
    if not os.path.exists(index_path(cache_dir)):
        return None

    index = load_index(cache_dir)
    stored = index.get("manifest_sha256")
    current = hash_file(manifest_path)
    if stored != current and not force:
        raise StaleCacheError(stored, current)
    return index
