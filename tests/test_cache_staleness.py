"""Staleness detection for the embedding cache index.

A cache whose stored manifest hash no longer matches the current manifest must be
refused (StaleCacheError); --force must override the refusal.

Runnable via `pytest tests/test_cache_staleness.py` or as a script.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import cache as C


def _make_manifest(tmp_path, text="line1\nline2\n"):
    p = os.path.join(tmp_path, "manifest.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def _write_index(cache_dir, manifest_hash):
    os.makedirs(cache_dir, exist_ok=True)
    C.write_index(cache_dir, {"manifest_sha256": manifest_hash, "records": []})


def test_no_index_is_fresh(tmp_path):
    manifest = _make_manifest(tmp_path)
    cache_dir = os.path.join(tmp_path, "cache")
    # no index.json yet -> nothing to resume, returns None, no raise
    assert C.check_resume(cache_dir, manifest) is None


def test_matching_hash_allows_resume(tmp_path):
    manifest = _make_manifest(tmp_path)
    cache_dir = os.path.join(tmp_path, "cache")
    _write_index(cache_dir, C.hash_file(manifest))

    index = C.check_resume(cache_dir, manifest)
    assert index is not None
    assert index["manifest_sha256"] == C.hash_file(manifest)


def test_mismatched_hash_is_refused(tmp_path):
    manifest = _make_manifest(tmp_path)
    cache_dir = os.path.join(tmp_path, "cache")
    _write_index(cache_dir, "deadbeef" * 8)  # wrong hash

    with pytest.raises(C.StaleCacheError):
        C.check_resume(cache_dir, manifest)


def test_force_overrides_stale(tmp_path):
    manifest = _make_manifest(tmp_path)
    cache_dir = os.path.join(tmp_path, "cache")
    _write_index(cache_dir, "deadbeef" * 8)  # wrong hash

    index = C.check_resume(cache_dir, manifest, force=True)  # must not raise
    assert index is not None


def test_hash_tracks_manifest_edits(tmp_path):
    manifest = _make_manifest(tmp_path, "original\n")
    cache_dir = os.path.join(tmp_path, "cache")
    _write_index(cache_dir, C.hash_file(manifest))

    # editing the manifest makes the cache stale
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("edited\n")
    with pytest.raises(C.StaleCacheError):
        C.check_resume(cache_dir, manifest)


if __name__ == "__main__":
    import tempfile

    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as d:
                fn(d)
            print(f"ok: {name}")
