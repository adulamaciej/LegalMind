import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.calibration import (
    TARGETS,
    _midranks,
    compute_calibration,
    compute_discrimination,
    error_detection_rate,
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


# --- _midranks ---

def test_midranks_without_ties():
    assert _midranks([80.0, 90.0, 85.0]) == [1.0, 3.0, 2.0]


def test_midranks_shares_average_rank_across_ties():
    # Two values tied for ranks 1 and 2 both get 1.5
    assert _midranks([85.0, 85.0]) == [1.5, 1.5]
    # Three-way tie for ranks 2, 3, 4 all get 3.0
    assert _midranks([70.0, 85.0, 85.0, 85.0]) == [1.0, 3.0, 3.0, 3.0]


# --- compute_discrimination: shape guarantees ---

DISCRIMINATION_KEYS = {
    "variant", "target", "n_scored", "n_excluded", "n_correct", "n_incorrect",
    "mean_confidence_when_correct", "mean_confidence_when_incorrect",
    "separation", "auroc", "undefined_reason",
}


def test_discrimination_empty_results_returns_full_schema():
    result = compute_discrimination([], target="violation")
    assert set(result) == DISCRIMINATION_KEYS
    assert result["separation"] is None
    assert result["auroc"] is None
    assert result["undefined_reason"] == "no scoreable cases"


def test_discrimination_undefined_when_every_case_correct():
    """Realistic at n=15 for the `exact` target, and must not raise."""
    result = compute_discrimination(correct_rows(5, 85), target="violation")
    assert set(result) == DISCRIMINATION_KEYS
    assert result["n_incorrect"] == 0
    assert result["separation"] is None
    assert result["auroc"] is None
    assert result["undefined_reason"] == "every scored case was correct"
    # The one group that does exist is still reported
    assert result["mean_confidence_when_correct"] == pytest.approx(0.85)
    assert result["mean_confidence_when_incorrect"] is None


def test_discrimination_undefined_when_every_case_incorrect():
    result = compute_discrimination(incorrect_rows(4, 60), target="violation")
    assert result["n_correct"] == 0
    assert result["separation"] is None
    assert result["undefined_reason"] == "every scored case was incorrect"
    assert result["mean_confidence_when_incorrect"] == pytest.approx(0.6)


def test_discrimination_counts_exclusions():
    rows = correct_rows(3, 90) + incorrect_rows(2, 70) + [make_row(50, "high")]
    result = compute_discrimination(rows, target="violation")
    assert result["n_scored"] == 5
    assert result["n_excluded"] == 1


def test_discrimination_rejects_unknown_target():
    with pytest.raises(ValueError, match="unknown target"):
        compute_discrimination(correct_rows(1, 85), target="nonsense")


def test_discrimination_rejects_old_results_file():
    legacy_row = {"case_id": 1, "exact_a": True, "flagged": False}
    with pytest.raises(ValueError, match="predates calibration support"):
        compute_discrimination([legacy_row], target="violation")


# --- compute_discrimination: separation ---

def test_separation_positive_when_confidence_tracks_correctness():
    rows = correct_rows(4, 90) + incorrect_rows(4, 60)
    result = compute_discrimination(rows, target="violation")
    assert result["separation"] == pytest.approx(0.3)
    assert result["auroc"] == pytest.approx(1.0)


def test_separation_negative_when_confidence_is_anticorrelated():
    """Confident exactly when wrong — triage built on this would be backwards."""
    rows = correct_rows(4, 60) + incorrect_rows(4, 90)
    result = compute_discrimination(rows, target="violation")
    assert result["separation"] == pytest.approx(-0.3)
    assert result["auroc"] == pytest.approx(0.0)


def test_separation_zero_when_confidence_is_constant():
    rows = correct_rows(5, 85) + incorrect_rows(5, 85)
    result = compute_discrimination(rows, target="violation")
    assert result["separation"] == pytest.approx(0.0)
    assert result["auroc"] == pytest.approx(0.5)


# --- compute_discrimination: AUROC against hand-computed values ---

def test_auroc_is_half_when_all_confidences_tie():
    rows = correct_rows(1, 85) + incorrect_rows(1, 85)
    assert compute_discrimination(rows, target="violation")["auroc"] == pytest.approx(0.5)


def test_auroc_with_partial_ties():
    """
    Correct = [90, 80], incorrect = [85]. Pairwise: 90 beats 85, 80 loses to 85
    — one win out of two, so AUROC is 0.5. Verifies the Mann-Whitney formula
    against a value that can be checked by hand.
    """
    rows = [
        make_row(0, 90, violation=True, truth_violation=True),
        make_row(1, 80, violation=True, truth_violation=True),
        make_row(2, 85, violation=True, truth_violation=False),
    ]
    assert compute_discrimination(rows, target="violation")["auroc"] == pytest.approx(0.5)


def test_auroc_gives_half_credit_for_a_tied_pair():
    """
    Correct = [90, 85], incorrect = [85]. One clear win plus one tie scored as
    half: (1 + 0.5) / 2 = 0.75.
    """
    rows = [
        make_row(0, 90, violation=True, truth_violation=True),
        make_row(1, 85, violation=True, truth_violation=True),
        make_row(2, 85, violation=True, truth_violation=False),
    ]
    assert compute_discrimination(rows, target="violation")["auroc"] == pytest.approx(0.75)


def test_auroc_is_base_rate_independent():
    """
    Why auroc, not ece, is used to compare targets: skewing the class balance
    while preserving the ranking must not move it.
    """
    balanced = correct_rows(3, 90) + incorrect_rows(3, 60)
    skewed = correct_rows(20, 90) + incorrect_rows(2, 60)
    auroc_balanced = compute_discrimination(balanced, target="violation")["auroc"]
    auroc_skewed = compute_discrimination(skewed, target="violation")["auroc"]
    assert auroc_balanced == pytest.approx(auroc_skewed)


def test_discrimination_differs_by_target():
    """Same verdicts, different targets — confidence can separate on one and not
    another, which is the whole reason for scoring three."""
    rows = [
        make_row(0, 95, predicted=["6"], truth=["6"]),
        make_row(1, 60, predicted=["6"], truth=["6", "8"]),
    ]
    # violation: both correct -> undefined
    assert compute_discrimination(rows, target="violation")["separation"] is None
    # exact: case 0 correct at 95, case 1 wrong at 60 -> clean separation
    assert compute_discrimination(rows, target="exact")["separation"] == pytest.approx(0.35)


# --- error_detection_rate ---

DETECTION_KEYS = {
    "variant", "target", "n_scored", "n_excluded", "n_errors",
    "undefined_reason", "budgets",
}

BUDGET_KEYS = {
    "review_fraction", "n_reviewed", "errors_caught", "detection_rate",
    "random_baseline", "lift", "boundary_ties",
}


def graded_rows(spec):
    """Rows from [(confidence, is_correct), ...], one case_id each."""
    return [
        make_row(i, confidence, violation=True, truth_violation=bool(ok))
        for i, (confidence, ok) in enumerate(spec)
    ]


def test_detection_empty_results_returns_full_schema():
    result = error_detection_rate([], target="violation")
    assert set(result) == DETECTION_KEYS
    assert result["n_errors"] == 0
    assert result["undefined_reason"] == "no scoreable cases"


def test_detection_undefined_when_there_are_no_errors():
    result = error_detection_rate(correct_rows(5, 85), target="violation")
    assert result["undefined_reason"] == "no errors to detect"
    assert all(b["detection_rate"] is None for b in result["budgets"])
    assert all(b["lift"] is None for b in result["budgets"])


def test_detection_budget_shape_is_stable():
    result = error_detection_rate(
        graded_rows([(60, 0), (90, 1)]), target="violation"
    )
    assert all(set(b) == BUDGET_KEYS for b in result["budgets"])


def test_detection_perfect_ranking_catches_every_error_in_budget():
    """4 errors at the bottom, 6 correct above; a 40% budget catches all of them."""
    spec = [(50, 0), (55, 0), (60, 0), (65, 0)] + [(90, 1)] * 6
    result = error_detection_rate(
        graded_rows(spec), target="violation", review_fractions=(0.4,)
    )
    budget = result["budgets"][0]
    assert result["n_errors"] == 4
    assert budget["n_reviewed"] == 4
    assert budget["errors_caught"] == 4
    assert budget["detection_rate"] == pytest.approx(1.0)
    assert budget["lift"] == pytest.approx(2.5)


def test_detection_worked_example_from_the_docstring():
    """10 cases, 4 errors, 30% budget = 3 cases holding 2 of the errors."""
    spec = [(55, 0), (60, 0), (65, 1), (70, 0), (75, 1),
            (80, 1), (85, 0), (90, 1), (95, 1), (100, 1)]
    result = error_detection_rate(
        graded_rows(spec), target="violation", review_fractions=(0.3,)
    )
    budget = result["budgets"][0]
    assert result["n_errors"] == 4
    assert budget["n_reviewed"] == 3
    assert budget["errors_caught"] == 2
    assert budget["detection_rate"] == pytest.approx(0.5)
    assert budget["random_baseline"] == pytest.approx(0.3)
    assert budget["lift"] == pytest.approx(0.5 / 0.3)


def test_detection_lift_below_one_when_confidence_is_anticorrelated():
    """Errors sit at the top, so reviewing the least-confident cases is worse
    than random — triage built on this confidence would be harmful."""
    spec = [(60, 1)] * 6 + [(90, 0), (92, 0), (94, 0), (96, 0)]
    result = error_detection_rate(
        graded_rows(spec), target="violation", review_fractions=(0.4,)
    )
    budget = result["budgets"][0]
    assert budget["errors_caught"] == 0
    assert budget["lift"] == pytest.approx(0.0)


def test_detection_lift_is_one_when_errors_are_evenly_spread():
    """Alternating correct/incorrect: sorting by confidence buys nothing."""
    spec = [(50, 0), (60, 1), (70, 0), (80, 1), (90, 0), (100, 1)]
    result = error_detection_rate(
        graded_rows(spec), target="violation", review_fractions=(1 / 3,)
    )
    budget = result["budgets"][0]
    assert budget["n_reviewed"] == 2
    assert budget["errors_caught"] == 1
    assert budget["lift"] == pytest.approx(1.0)


def test_detection_budget_rounds_up_so_it_is_never_empty():
    """10% of 15 cases is 1.5 — rounding down would review nobody."""
    spec = [(50 + i, i % 3 == 0) for i in range(15)]
    result = error_detection_rate(
        graded_rows(spec), target="violation", review_fractions=(0.1,)
    )
    assert result["budgets"][0]["n_reviewed"] == 2


def test_detection_budget_never_exceeds_the_sample():
    result = error_detection_rate(
        graded_rows([(60, 0), (90, 1)]), target="violation", review_fractions=(1.0,)
    )
    assert result["budgets"][0]["n_reviewed"] == 2
    assert result["budgets"][0]["detection_rate"] == pytest.approx(1.0)


def test_detection_flags_ties_across_the_budget_boundary():
    """Four cases tied at 85 with a 2-case budget: which two is arbitrary, so the
    number must not be presented as reproducible."""
    spec = [(85, 0), (85, 0), (85, 1), (85, 1), (95, 1)]
    result = error_detection_rate(
        graded_rows(spec), target="violation", review_fractions=(0.4,)
    )
    assert result["budgets"][0]["boundary_ties"] is True


def test_detection_does_not_flag_ties_when_boundary_is_clean():
    spec = [(60, 0), (60, 0), (90, 1), (90, 1)]
    result = error_detection_rate(
        graded_rows(spec), target="violation", review_fractions=(0.5,)
    )
    assert result["budgets"][0]["boundary_ties"] is False


def test_detection_is_order_independent():
    """Same cases shuffled must give the same answer."""
    spec = [(55, 0), (60, 0), (65, 1), (70, 0), (75, 1), (80, 1)]
    forward = error_detection_rate(graded_rows(spec), target="violation")
    backward = error_detection_rate(
        list(reversed(graded_rows(spec))), target="violation"
    )
    assert forward["budgets"] == backward["budgets"]


def test_detection_budgets_are_sorted_ascending():
    result = error_detection_rate(
        graded_rows([(60, 0), (70, 1), (80, 0), (90, 1)]),
        target="violation",
        review_fractions=(0.5, 0.25),
    )
    fractions = [b["review_fraction"] for b in result["budgets"]]
    assert fractions == sorted(fractions)


def test_detection_counts_exclusions():
    rows = graded_rows([(60, 0), (90, 1)]) + [make_row(99, "high")]
    result = error_detection_rate(rows, target="violation")
    assert result["n_scored"] == 2
    assert result["n_excluded"] == 1


def test_detection_survives_rows_without_a_case_id():
    """case_id is only a tiebreaker, so a row missing it must not raise."""
    rows = graded_rows([(85, 0), (85, 1)])
    for row in rows:
        del row["case_id"]
    result = error_detection_rate(rows, target="violation", review_fractions=(0.5,))
    assert result["n_scored"] == 2


def test_detection_rejects_invalid_review_fraction():
    rows = graded_rows([(60, 0), (90, 1)])
    with pytest.raises(ValueError, match="review_fractions"):
        error_detection_rate(rows, target="violation", review_fractions=(0.0,))
    with pytest.raises(ValueError, match="review_fractions"):
        error_detection_rate(rows, target="violation", review_fractions=(1.5,))


def test_detection_rejects_unknown_target():
    with pytest.raises(ValueError, match="unknown target"):
        error_detection_rate(correct_rows(1, 85), target="nonsense")


def test_detection_rejects_old_results_file():
    with pytest.raises(ValueError, match="predates calibration support"):
        error_detection_rate([{"case_id": 1, "exact_a": True}], target="violation")


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
