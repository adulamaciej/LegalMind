import sys
import os
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.results_io import load_results, results_filename, save_results


FIXED_MOMENT = datetime(2026, 8, 4, 14, 22, 1, tzinfo=timezone.utc)

SAMPLE_META = {
    "model": "claude-haiku-4-5-20251001",
    "sampling": "sdk default (temperature unset)",
    "n_cases_requested": 15,
    "n_cases_completed": 14,
    "seed": 42,
    "failures": [{"case_id": 412, "error": "JSONDecodeError: Expecting value"}],
}

SAMPLE_RESULTS = [
    {
        "case_id": 7,
        "confidence_a": 85, "violation_a": True, "predicted_a": ["6"],
        "confidence_b": 60, "violation_b": False, "predicted_b": [],
        "truth": ["6"], "truth_violation": True,
    }
]


def test_filename_has_no_characters_windows_rejects():
    """A colon is legal in an ISO timestamp and illegal in a Windows filename."""
    name = results_filename(FIXED_MOMENT)
    assert ":" not in name
    for illegal in '<>:"/\\|?*':
        assert illegal not in name


def test_filename_encodes_the_moment():
    assert results_filename(FIXED_MOMENT) == "eval_20260804_142201.json"


def test_save_and_load_roundtrip(tmp_path):
    path = save_results(SAMPLE_RESULTS, SAMPLE_META, tmp_path, FIXED_MOMENT)
    results, meta = load_results(path)
    assert results == SAMPLE_RESULTS
    assert meta == SAMPLE_META


def test_save_creates_the_directory(tmp_path):
    target = tmp_path / "nested" / "results"
    path = save_results(SAMPLE_RESULTS, SAMPLE_META, target, FIXED_MOMENT)
    assert path.exists()
    assert path.parent == target


def test_saved_file_is_stable_across_runs(tmp_path):
    """Two batches with the same content must diff clean, so key order is fixed."""
    first = save_results(SAMPLE_RESULTS, SAMPLE_META, tmp_path / "a", FIXED_MOMENT)
    second = save_results(SAMPLE_RESULTS, SAMPLE_META, tmp_path / "b", FIXED_MOMENT)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_saved_file_preserves_the_failure_log(tmp_path):
    """A systematic failure biases the sample, so which cases dropped out has to
    survive into the file rather than only reaching the console."""
    path = save_results(SAMPLE_RESULTS, SAMPLE_META, tmp_path, FIXED_MOMENT)
    _, meta = load_results(path)
    assert meta["failures"][0]["case_id"] == 412


def test_saved_file_records_what_makes_a_run_reproducible(tmp_path):
    path = save_results(SAMPLE_RESULTS, SAMPLE_META, tmp_path, FIXED_MOMENT)
    _, meta = load_results(path)
    for field in ("model", "sampling", "seed", "n_cases_requested"):
        assert field in meta


def test_save_handles_an_empty_batch(tmp_path):
    path = save_results([], {"n_cases_completed": 0}, tmp_path, FIXED_MOMENT)
    results, meta = load_results(path)
    assert results == []
    assert meta["n_cases_completed"] == 0


def test_load_rejects_a_file_that_is_not_a_results_batch(tmp_path):
    stray = tmp_path / "stray.json"
    stray.write_text('[{"case_id": 1}]', encoding="utf-8")
    with pytest.raises(ValueError, match="not an evaluation results file"):
        load_results(stray)


def test_load_tolerates_a_missing_meta_block(tmp_path):
    partial = tmp_path / "partial.json"
    partial.write_text('{"results": []}', encoding="utf-8")
    results, meta = load_results(partial)
    assert results == []
    assert meta == {}


def test_saved_file_is_utf8(tmp_path):
    """Reasoning text carries non-ASCII, and the default encoding on Windows
    is not UTF-8."""
    rows = [{"case_id": 1, "note": "naruszenie art. 6 — rzetelny proces"}]
    path = save_results(rows, {}, tmp_path, FIXED_MOMENT)
    assert "rzetelny proces" in path.read_text(encoding="utf-8")
