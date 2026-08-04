import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.calibration import (
    TARGETS,
    compute_calibration,
    normalize_confidence,
)


def make_row(
    case_id=0,
    confidence=85,
    predicted=(),
    truth=(),
    violation=None,
    truth_violation=None,
):
    """Build one result row for variant 'a'."""
    return {
        "case_id": case_id,
        "confidence_a": confidence,
        "predicted_a": list(predicted),
        "violation_a": bool(predicted) if violation is None else violation,
        "truth": list(truth),
        "truth_violation": bool(truth) if truth_violation is None else truth_violation,
    }


def correct_rows(count, confidence, start=0):
    """Rows where the violation call is right."""
    return [
        make_row(start + i, confidence, violation=True, truth_violation=True)
        for i in range(count)
    ]


def incorrect_rows(count, confidence, start=1000):
    """Rows where the violation call is wrong."""
    return [
        make_row(start + i, confidence, violation=True, truth_violation=False)
        for i in range(count)
    ]


# --- normalize_confidence ---

def test_normalize_confidence_accepts_percent():
    assert normalize_confidence(85) == 85.0


def test_normalize_confidence_converts_fraction():
    assert normalize_confidence(0.85) == 85.0


def test_normalize_confidence_accepts_numeric_string():
    assert normalize_confidence("85") == 85.0


def test_normalize_confidence_accepts_boundaries():
    assert normalize_confidence(0) == 0.0
    assert normalize_confidence(100) == 100.0


def test_normalize_confidence_rejects_out_of_range():
    assert normalize_confidence(150) is None
    assert normalize_confidence(-5) is None


def test_normalize_confidence_rejects_garbage():
    assert normalize_confidence("high") is None
    assert normalize_confidence(None) is None
    assert normalize_confidence([85]) is None


def test_normalize_confidence_rejects_bool():
    # bool subclasses int; True must not silently become 100%
    assert normalize_confidence(True) is None
    assert normalize_confidence(False) is None


def test_normalize_confidence_rejects_nan():
    assert normalize_confidence(float("nan")) is None


# --- compute_calibration: shape guarantees ---

EXPECTED_KEYS = {
    "variant", "target", "n_scored", "n_excluded", "brier", "ece",
    "mean_confidence", "accuracy", "overconfidence", "no_discrimination",
    "uninformative_target", "distinct_confidence_values", "bins",
}


def test_empty_results_returns_full_schema_without_crashing():
    result = compute_calibration([], target="violation")
    assert set(result) == EXPECTED_KEYS
    assert result["n_scored"] == 0
    assert result["brier"] is None


def test_single_case_does_not_crash():
    # n_cases defaults to 1 in run_evaluation, so this is a real path
    result = compute_calibration(correct_rows(1, 90), target="violation")
    assert result["n_scored"] == 1
    assert result["accuracy"] == 1.0


def test_no_discrimination_still_returns_full_schema():
    """Constant confidence must not change the return shape."""
    result = compute_calibration(
        correct_rows(6, 85) + incorrect_rows(4, 85), target="violation"
    )
    assert set(result) == EXPECTED_KEYS
    assert result["no_discrimination"] is True
    assert result["brier"] is not None
    assert result["ece"] is not None


def test_no_discrimination_ece_equals_absolute_overconfidence():
    """With one populated bin, ECE carries nothing beyond |overconfidence|."""
    result = compute_calibration(
        correct_rows(6, 85) + incorrect_rows(4, 85), target="violation"
    )
    assert result["ece"] == pytest.approx(abs(result["overconfidence"]))


def test_uninformative_target_flagged_when_all_correct():
    result = compute_calibration(correct_rows(5, 80), target="violation")
    assert result["uninformative_target"] is True
    assert result["accuracy"] == 1.0


def test_varied_confidence_not_flagged_as_degenerate():
    rows = correct_rows(3, 70) + incorrect_rows(3, 90)
    result = compute_calibration(rows, target="violation")
    assert result["no_discrimination"] is False
    assert result["uninformative_target"] is False


# --- compute_calibration: metric correctness ---

def test_perfect_calibration_has_zero_ece():
    # 80 of 100 correct, all claiming 80%
    rows = correct_rows(80, 80) + incorrect_rows(20, 80)
    result = compute_calibration(rows, target="violation")
    assert result["accuracy"] == pytest.approx(0.8)
    assert result["mean_confidence"] == pytest.approx(0.8)
    assert result["ece"] == pytest.approx(0.0)
    assert result["overconfidence"] == pytest.approx(0.0)


def test_overconfident_model_has_positive_overconfidence():
    rows = correct_rows(5, 95) + incorrect_rows(5, 95)
    result = compute_calibration(rows, target="violation")
    assert result["overconfidence"] == pytest.approx(0.45)
    assert result["ece"] == pytest.approx(0.45)
    assert result["brier"] == pytest.approx(0.4525)


def test_underconfident_model_has_negative_overconfidence():
    rows = correct_rows(9, 40) + incorrect_rows(1, 40)
    result = compute_calibration(rows, target="violation")
    assert result["overconfidence"] == pytest.approx(-0.5)
    assert result["ece"] == pytest.approx(0.5)


def test_brier_is_zero_for_a_perfect_forecaster():
    rows = correct_rows(3, 100) + incorrect_rows(3, 0)
    result = compute_calibration(rows, target="violation")
    assert result["brier"] == pytest.approx(0.0)


# --- compute_calibration: binning decisions ---

def test_bin_uses_mean_confidence_not_midpoint():
    """
    Pins the choice of mean-confidence over bin-midpoint in the ECE formula.
    Confidences 82/84/86 land in the [0.8, 1.0] bin whose midpoint is 0.9;
    their mean is 0.84. With all three correct, ECE must be |1.0 - 0.84|,
    not |1.0 - 0.9|.
    """
    rows = [
        make_row(0, 82, violation=True, truth_violation=True),
        make_row(1, 84, violation=True, truth_violation=True),
        make_row(2, 86, violation=True, truth_violation=True),
    ]
    result = compute_calibration(rows, target="violation", n_bins=5)
    assert result["bins"][4]["mean_confidence"] == pytest.approx(0.84)
    assert result["ece"] == pytest.approx(0.16)


def test_bins_are_stable_and_sorted():
    result = compute_calibration(correct_rows(2, 85), target="violation", n_bins=5)
    bins = result["bins"]
    assert len(bins) == 5
    assert [b["lo"] for b in bins] == sorted(b["lo"] for b in bins)
    # Empty bins are still emitted, so two runs diff cleanly
    assert bins[0]["n"] == 0
    assert bins[4]["n"] == 2


def test_confidence_of_100_lands_in_last_bin():
    result = compute_calibration(correct_rows(1, 100), target="violation", n_bins=5)
    assert result["bins"][-1]["n"] == 1


def test_invalid_n_bins_rejected():
    with pytest.raises(ValueError, match="n_bins"):
        compute_calibration(correct_rows(1, 85), target="violation", n_bins=0)


# --- compute_calibration: exclusions and schema ---

def test_excluded_cases_are_counted_not_dropped_silently():
    rows = correct_rows(7, 85) + [
        make_row(90, "high"),
        make_row(91, None),
        make_row(92, 150),
    ]
    result = compute_calibration(rows, target="violation")
    assert result["n_scored"] == 7
    assert result["n_excluded"] == 3


def test_all_cases_excluded_returns_full_schema():
    result = compute_calibration([make_row(0, "high")], target="violation")
    assert set(result) == EXPECTED_KEYS
    assert result["n_scored"] == 0
    assert result["n_excluded"] == 1


def test_old_results_file_raises_clear_error():
    legacy_row = {"case_id": 1, "exact_a": True, "partial_a": True, "flagged": False}
    with pytest.raises(ValueError, match="predates calibration support"):
        compute_calibration([legacy_row], target="violation")


def test_schema_check_scans_every_row():
    rows = correct_rows(3, 85)
    del rows[2]["truth_violation"]
    with pytest.raises(ValueError, match="row 2"):
        compute_calibration(rows, target="violation")


def test_unknown_target_rejected():
    with pytest.raises(ValueError, match="unknown target"):
        compute_calibration(correct_rows(1, 85), target="nonsense")


# --- compute_calibration: targets are genuinely different ---

def test_targets_disagree_on_the_same_verdicts():
    """
    Predicting ["6"] when the truth is ["6", "8"] is a correct violation call,
    a wrong exact match, and a correct partial match. Each target must score it
    differently — otherwise scoring three targets is pointless.
    """
    rows = [make_row(i, 85, predicted=["6"], truth=["6", "8"]) for i in range(4)]
    accuracies = {
        target: compute_calibration(rows, target=target)["accuracy"]
        for target in TARGETS
    }
    assert accuracies["violation"] == 1.0
    assert accuracies["exact"] == 0.0
    assert accuracies["any_correct"] == 1.0


def test_any_correct_matches_empty_ground_truth_convention():
    """Mirrors partial_match in evaluation.py: predicting nothing when there is
    nothing to predict counts as correct."""
    empty_hit = [make_row(0, 85, predicted=[], truth=[])]
    empty_miss = [make_row(1, 85, predicted=["6"], truth=[])]
    assert compute_calibration(empty_hit, target="any_correct")["accuracy"] == 1.0
    assert compute_calibration(empty_miss, target="any_correct")["accuracy"] == 0.0


def test_distinct_confidence_values_reported_as_sorted_percentages():
    rows = correct_rows(1, 90) + correct_rows(1, 70, start=5) + correct_rows(1, 90, start=9)
    result = compute_calibration(rows, target="violation")
    assert result["distinct_confidence_values"] == [70.0, 90.0]


def test_variant_b_scored_independently():
    rows = [
        {
            "case_id": 0,
            "confidence_a": 90, "predicted_a": ["6"], "violation_a": True,
            "confidence_b": 60, "predicted_b": [], "violation_b": False,
            "truth": ["6"], "truth_violation": True,
        }
    ]
    cal_a = compute_calibration(rows, variant="a", target="violation")
    cal_b = compute_calibration(rows, variant="b", target="violation")
    assert cal_a["accuracy"] == 1.0
    assert cal_b["accuracy"] == 0.0
    assert cal_a["mean_confidence"] == pytest.approx(0.9)
    assert cal_b["mean_confidence"] == pytest.approx(0.6)
