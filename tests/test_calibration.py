import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.calibration import (
    TARGETS,
    TRACKED_TARGET_MIN_AUROC,
    TRACKED_TARGET_MIN_MARGIN,
    _midranks,
    _tracked_target,
    compare_variants_paired,
    compute_calibration,
    compute_discrimination,
    error_detection_rate,
    format_calibration_report,
    format_uncertainty_signals_report,
    normalize_confidence,
    sign_test,
    signal_error_lift,
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


# --- sign_test, against hand-computed exact values ---

def test_sign_test_symmetric_split_is_not_significant():
    # extreme = 5, tail = C(10,5..10) = 638, p = 2 * 638 / 1024 -> capped at 1.0
    assert sign_test(5, 10) == pytest.approx(1.0)


def test_sign_test_all_wins():
    # extreme = 5, tail = C(5,5) = 1, p = 2 * 1 / 32
    assert sign_test(5, 5) == pytest.approx(0.0625)


def test_sign_test_all_losses_is_symmetric():
    assert sign_test(0, 5) == pytest.approx(sign_test(5, 5))


def test_sign_test_nine_of_fourteen():
    # tail = C(14,9..14) = 2002+1001+364+91+14+1 = 3473; p = 2 * 3473 / 16384
    assert sign_test(9, 14) == pytest.approx(2 * 3473 / 16384)
    assert sign_test(9, 14) == pytest.approx(0.4239502, abs=1e-6)


def test_sign_test_single_case_can_never_be_significant():
    assert sign_test(1, 1) == pytest.approx(1.0)
    assert sign_test(0, 1) == pytest.approx(1.0)


def test_sign_test_undefined_without_decisive_cases():
    assert sign_test(0, 0) is None


def test_sign_test_rejects_impossible_win_count():
    with pytest.raises(ValueError, match="n_wins"):
        sign_test(6, 5)
    with pytest.raises(ValueError, match="n_wins"):
        sign_test(-1, 5)


def test_sign_test_never_exceeds_one():
    for n in range(1, 21):
        for wins in range(n + 1):
            assert 0.0 < sign_test(wins, n) <= 1.0


# --- compare_variants_paired ---

PAIRED_KEYS = {
    "target", "n_paired", "n_scored_a", "n_scored_b", "brier_a", "brier_b",
    "mean_delta", "sd_delta", "min_delta", "max_delta", "favors_a", "favors_b",
    "ties", "n_decisive", "sign_test_p", "undefined_reason", "per_case",
}


def ab_row(case_id, confidence_a, correct_a, confidence_b, correct_b):
    """
    One case scored in both variants.

    Correctness is expressed for the `violation` target only: ground truth is
    always "a violation occurred", and a variant is made wrong by having it rule
    the opposite. Tests that need a per-target difference build their row inline.
    """
    return {
        "case_id": case_id,
        "confidence_a": confidence_a,
        "violation_a": bool(correct_a),
        "predicted_a": ["6"],
        "confidence_b": confidence_b,
        "violation_b": bool(correct_b),
        "predicted_b": ["6"],
        "truth_violation": True,
        "truth": ["6"],
    }


def test_paired_empty_results_returns_full_schema():
    result = compare_variants_paired([], target="violation")
    assert set(result) == PAIRED_KEYS
    assert result["n_paired"] == 0
    assert result["sign_test_p"] is None
    assert result["undefined_reason"] == "no cases scoreable in both variants"


def test_paired_mean_delta_equals_brier_difference():
    """The pairing buys the distribution, not the point estimate — this identity
    is why mean_delta must not be presented as new information."""
    rows = [
        ab_row(0, 90, True, 60, True),
        ab_row(1, 80, False, 95, False),
        ab_row(2, 70, True, 70, False),
    ]
    result = compare_variants_paired(rows, target="violation")
    assert result["mean_delta"] == pytest.approx(result["brier_a"] - result["brier_b"])


def test_paired_favours_the_variant_with_lower_brier():
    # A confident and right, B unconfident and right -> A has lower Brier
    rows = [ab_row(0, 95, True, 55, True)]
    result = compare_variants_paired(rows, target="violation")
    assert result["favors_a"] == 1
    assert result["favors_b"] == 0
    assert result["mean_delta"] < 0


def test_paired_counts_split_across_wins_losses_and_ties():
    rows = [
        ab_row(0, 95, True, 55, True),    # A better
        ab_row(1, 95, True, 55, True),    # A better
        ab_row(2, 55, True, 95, True),    # B better
        ab_row(3, 80, True, 80, True),    # tie
        ab_row(4, 70, False, 70, False),  # tie
    ]
    result = compare_variants_paired(rows, target="violation")
    assert (result["favors_a"], result["favors_b"], result["ties"]) == (2, 1, 2)
    assert result["n_paired"] == 5


def test_paired_ties_excluded_from_sign_test_denominator():
    rows = [
        ab_row(0, 95, True, 55, True),    # A better
        ab_row(1, 95, True, 55, True),    # A better
        ab_row(2, 80, True, 80, True),    # tie
        ab_row(3, 80, True, 80, True),    # tie
        ab_row(4, 80, True, 80, True),    # tie
    ]
    result = compare_variants_paired(rows, target="violation")
    assert result["n_decisive"] == 2
    assert result["sign_test_p"] == pytest.approx(sign_test(2, 2))


def test_paired_sign_test_undefined_when_every_case_ties():
    """Brier comparison still holds; only the test is undefined."""
    rows = [ab_row(i, 80, True, 80, True) for i in range(3)]
    result = compare_variants_paired(rows, target="violation")
    assert result["n_decisive"] == 0
    assert result["sign_test_p"] is None
    assert result["mean_delta"] == pytest.approx(0.0)
    assert result["undefined_reason"] is None


def test_paired_uses_only_cases_scoreable_in_both_variants():
    """A case usable in A but not B must drop out of the pairing, and the
    per-variant counts must expose the discrepancy."""
    rows = [
        ab_row(0, 90, True, 60, True),
        ab_row(1, 90, True, 60, True),
    ]
    rows[1]["confidence_b"] = "unusable"
    result = compare_variants_paired(rows, target="violation")
    assert result["n_paired"] == 1
    assert result["n_scored_a"] == 2
    assert result["n_scored_b"] == 1


def test_paired_brier_computed_on_the_paired_subset():
    """Numbers in one report section must reconcile, so brier_a here is over the
    paired cases only — not over everything variant A could score."""
    rows = [
        ab_row(0, 100, True, 100, True),   # brier_a = 0
        ab_row(1, 0, False, 0, False),     # brier_a = 0, but dropped below
    ]
    rows[1]["confidence_b"] = None
    result = compare_variants_paired(rows, target="violation")
    assert result["n_paired"] == 1
    assert result["brier_a"] == pytest.approx(0.0)


def test_paired_spread_reported():
    rows = [
        ab_row(0, 95, True, 55, True),
        ab_row(1, 55, True, 95, True),
    ]
    result = compare_variants_paired(rows, target="violation")
    assert result["min_delta"] < 0 < result["max_delta"]
    assert result["sd_delta"] > 0


def test_paired_sd_undefined_for_a_single_case():
    result = compare_variants_paired([ab_row(0, 90, True, 60, True)], target="violation")
    assert result["n_paired"] == 1
    assert result["sd_delta"] is None


def test_paired_per_case_rows_carry_case_ids():
    rows = [ab_row(7, 90, True, 60, True), ab_row(3, 80, False, 80, True)]
    result = compare_variants_paired(rows, target="violation")
    assert [case["case_id"] for case in result["per_case"]] == [7, 3]


def test_paired_respects_the_target():
    """Variant A right on `violation` but wrong on `exact` must flip the winner."""
    row = {
        "case_id": 0,
        "confidence_a": 95, "violation_a": True, "predicted_a": ["6"],
        "confidence_b": 95, "violation_b": True, "predicted_b": ["6", "8"],
        "truth_violation": True, "truth": ["6", "8"],
    }
    on_violation = compare_variants_paired([row], target="violation")
    on_exact = compare_variants_paired([row], target="exact")
    assert on_violation["ties"] == 1          # both correct, same confidence
    assert on_exact["favors_b"] == 1          # only B named the right set


def test_paired_rejects_unknown_target():
    with pytest.raises(ValueError, match="unknown target"):
        compare_variants_paired([ab_row(0, 90, True, 60, True)], target="nonsense")


def test_paired_rejects_old_results_file():
    with pytest.raises(ValueError, match="predates calibration support"):
        compare_variants_paired([{"case_id": 1, "exact_a": True}], target="violation")


def test_paired_requires_variant_b_fields():
    row = ab_row(0, 90, True, 60, True)
    del row["confidence_b"]
    with pytest.raises(ValueError, match="confidence_b"):
        compare_variants_paired([row], target="violation")


# --- _tracked_target: the rule for naming what confidence follows ---

def discrimination_stub(auroc):
    return {"auroc": auroc}


def test_tracked_target_names_a_clear_winner():
    target, reason, _ = _tracked_target({
        "violation": discrimination_stub(0.82),
        "exact": discrimination_stub(0.55),
        "any_correct": discrimination_stub(0.60),
    })
    assert target == "violation"
    assert "0.82" in reason


def test_tracked_target_refuses_below_the_auroc_floor():
    """A target can lead and still carry no usable signal."""
    below = TRACKED_TARGET_MIN_AUROC - 0.05
    target, reason, _ = _tracked_target({
        "violation": discrimination_stub(below),
        "exact": discrimination_stub(0.50),
        "any_correct": discrimination_stub(0.45),
    })
    assert target is None
    assert "floor" in reason


def test_tracked_target_refuses_when_the_margin_is_too_thin():
    """At n=15 a few hundredths of auroc is noise, not a ranking."""
    target, reason, _ = _tracked_target({
        "violation": discrimination_stub(0.80),
        "exact": discrimination_stub(0.80 - TRACKED_TARGET_MIN_MARGIN / 2),
        "any_correct": discrimination_stub(0.50),
    })
    assert target is None
    assert "too close to call" in reason


def test_tracked_target_refuses_when_nothing_is_rankable():
    target, reason, _ = _tracked_target({
        target: discrimination_stub(None) for target in TARGETS
    })
    assert target is None
    assert "rank" in reason


def test_tracked_target_ignores_unrankable_targets():
    target, _, n_rankable = _tracked_target({
        "violation": discrimination_stub(None),
        "exact": discrimination_stub(0.90),
        "any_correct": discrimination_stub(None),
    })
    assert target == "exact"
    assert n_rankable == 1


def test_tracked_target_reports_the_margin_when_a_comparison_happened():
    _, reason, n_rankable = _tracked_target({
        "violation": discrimination_stub(0.90),
        "exact": discrimination_stub(0.70),
        "any_correct": discrimination_stub(None),
    })
    assert n_rankable == 2
    assert "clear of 'exact'" in reason


def test_report_does_not_claim_a_win_when_only_one_target_is_rankable():
    """The other targets were unmeasurable, not worse — saying the winner
    'tracks most closely' would imply a comparison that never happened."""
    rows = mixed_ab_rows()
    report = format_calibration_report(rows)
    assert "was the only rankable target" in report
    assert "most closely" not in report


def test_report_pluralises_the_review_budget():
    rows = mixed_ab_rows()
    report = format_calibration_report(rows)
    assert "(1 case)" in report
    assert "(1 cases)" not in report


# --- format_calibration_report ---

def mixed_ab_rows():
    """A spread of confidences and outcomes that leaves most metrics defined."""
    spec = [
        (95, True, 60, True), (90, True, 65, False), (88, True, 70, True),
        (85, False, 75, False), (80, True, 80, True), (75, False, 85, False),
        (70, True, 90, True), (65, False, 95, False),
    ]
    return [
        ab_row(i, conf_a, ok_a, conf_b, ok_b)
        for i, (conf_a, ok_a, conf_b, ok_b) in enumerate(spec)
    ]


def test_report_handles_empty_results():
    report = format_calibration_report([])
    assert "No results to score" in report


def test_report_covers_every_target():
    report = format_calibration_report(mixed_ab_rows())
    for target in TARGETS:
        assert f"target: {target}" in report


def test_report_is_deterministic():
    """Two runs on the same results must diff clean, so nothing time-varying
    or dict-order-dependent may leak into the text."""
    rows = mixed_ab_rows()
    assert format_calibration_report(rows) == format_calibration_report(rows)


def test_report_labels_paired_and_unpaired_sections():
    """The distinction carries the evidential weight, so it must be visible."""
    report = format_calibration_report(mixed_ab_rows())
    assert "paired — same cases" in report
    assert "calibration (unpaired)" in report
    assert "discrimination (unpaired)" in report


def test_report_warns_when_accuracy_gap_confounds_ece():
    rows = [ab_row(i, 85, True, 85, i < 2) for i in range(10)]
    report = format_calibration_report(rows)
    assert "accuracy differs by" in report
    assert "read overconfidence instead" in report


def test_report_omits_accuracy_warning_when_variants_agree():
    rows = [ab_row(i, 85, i % 2 == 0, 85, i % 2 == 0) for i in range(10)]
    report = format_calibration_report(rows)
    assert "accuracy differs by" not in report


def test_report_warns_about_constant_confidence():
    rows = [ab_row(i, 85, i % 2 == 0, 85, i % 2 == 0) for i in range(8)]
    report = format_calibration_report(rows)
    assert "confidence is constant" in report
    assert "carries no extra information" in report


def test_report_warns_about_uninformative_target():
    rows = [ab_row(i, 70 + i, True, 70 + i, True) for i in range(6)]
    report = format_calibration_report(rows)
    assert "every scored case had the same outcome" in report


def test_report_renders_undefined_metrics_as_a_dash_not_a_number():
    """All correct means discrimination is undefined; it must not print 0.000."""
    rows = [ab_row(i, 70 + i, True, 70 + i, True) for i in range(6)]
    report = format_calibration_report(rows)
    assert "undefined — every scored case was correct" in report


def test_report_states_the_tracking_conclusion():
    report = format_calibration_report(mixed_ab_rows())
    assert "what does confidence track?" in report
    assert "decided on auroc, never ece" in report


def test_report_survives_a_single_case():
    report = format_calibration_report([ab_row(0, 85, True, 85, False)])
    assert "target: violation" in report


def test_report_survives_rows_with_unusable_confidence():
    rows = mixed_ab_rows()
    rows[0]["confidence_a"] = "high"
    rows[1]["confidence_b"] = None
    report = format_calibration_report(rows)
    assert "excluded:" in report


def test_report_flags_tied_review_budgets():
    rows = [ab_row(i, 85, i < 4, 85, i < 4) for i in range(8)]
    report = format_calibration_report(rows)
    assert "!ties" in report


# --- signal_error_lift: is each uncertainty flag worth anything? ---

def signal_row(case_id, fired, correct, confidence=85, signal="low_confidence"):
    """A case for the `exact` target, with one uncertainty flag set or clear."""
    row = {
        "case_id": case_id,
        "confidence_a": confidence,
        "predicted_a": ["6"] if correct else ["8"],
        "violation_a": True,
        "truth": ["6"],
        "truth_violation": True,
        "low_confidence_a": False,
        "unsupported_article_6_a": False,
        "hallucinated_a": [],
    }
    row[f"{signal}_a"] = ["13"] if signal == "hallucinated" and fired else bool(fired)
    return row


def test_signal_lift_above_one_when_the_flag_marks_bad_verdicts():
    """4 flagged and all wrong, 6 clear and all right: the flag is perfect."""
    rows = [signal_row(i, True, False) for i in range(4)]
    rows += [signal_row(10 + i, False, True) for i in range(6)]
    stats = signal_error_lift(rows, "low_confidence")
    assert stats["n_fired"] == 4
    assert stats["error_rate_when_fired"] == pytest.approx(1.0)
    assert stats["error_rate_overall"] == pytest.approx(0.4)
    assert stats["lift"] == pytest.approx(2.5)
    assert stats["coverage"] == pytest.approx(1.0)


def test_signal_lift_of_one_when_the_flag_is_uninformative():
    """Errors split evenly across flagged and unflagged."""
    rows = [
        signal_row(0, True, False), signal_row(1, True, True),
        signal_row(2, False, False), signal_row(3, False, True),
    ]
    stats = signal_error_lift(rows, "low_confidence")
    assert stats["lift"] == pytest.approx(1.0)


def test_signal_lift_below_one_when_the_flag_points_the_wrong_way():
    """The flag fires on the correct verdicts — following it wastes review time."""
    rows = [signal_row(i, True, True) for i in range(3)]
    rows += [signal_row(10 + i, False, False) for i in range(3)]
    stats = signal_error_lift(rows, "low_confidence")
    assert stats["error_rate_when_fired"] == pytest.approx(0.0)
    assert stats["lift"] == pytest.approx(0.0)


def test_signal_coverage_is_reported_separately_from_lift():
    """A flag can be perfectly precise and still miss most errors."""
    rows = [signal_row(0, True, False)]
    rows += [signal_row(10 + i, False, False) for i in range(3)]
    stats = signal_error_lift(rows, "low_confidence")
    assert stats["error_rate_when_fired"] == pytest.approx(1.0)
    assert stats["coverage"] == pytest.approx(0.25)


def test_signal_undefined_when_it_never_fires():
    rows = [signal_row(i, False, i % 2 == 0) for i in range(4)]
    stats = signal_error_lift(rows, "low_confidence")
    assert stats["n_fired"] == 0
    assert stats["undefined_reason"] == "the signal never fired"


def test_signal_undefined_when_there_are_no_errors():
    rows = [signal_row(i, i < 2, True) for i in range(4)]
    stats = signal_error_lift(rows, "low_confidence")
    assert stats["undefined_reason"] == "no errors to point at"


def test_signal_reports_confidence_on_each_side():
    """Answers a question the combined flag could not: does the model sound
    more or less certain when it invents an article code?"""
    rows = [signal_row(0, True, False, confidence=95, signal="hallucinated")]
    rows += [signal_row(10 + i, False, True, confidence=70, signal="hallucinated")
             for i in range(3)]
    stats = signal_error_lift(rows, "hallucinated")
    assert stats["mean_confidence_when_fired"] == pytest.approx(0.95)
    assert stats["mean_confidence_when_quiet"] == pytest.approx(0.70)


def test_signal_treats_a_nonempty_hallucination_list_as_fired():
    rows = [signal_row(0, True, False, signal="hallucinated")]
    rows += [signal_row(10 + i, False, True, signal="hallucinated") for i in range(3)]
    stats = signal_error_lift(rows, "hallucinated")
    assert stats["n_fired"] == 1


def test_signal_reports_a_clear_reason_on_an_old_results_file():
    legacy = [{"case_id": 1, "exact_a": True, "flagged": False}]
    stats = signal_error_lift(legacy, "low_confidence")
    assert stats["n_fired"] is None
    assert "predates the split-out signals" in stats["undefined_reason"]


def test_signal_handles_an_empty_batch():
    stats = signal_error_lift([], "low_confidence")
    assert stats["n_total"] == 0
    assert stats["undefined_reason"] == "the signal never fired"


# --- format_uncertainty_signals_report ---

def test_uncertainty_report_covers_all_three_signals_separately():
    """The point of the rewrite: each flag scored on its own, not OR-ed."""
    rows = [signal_row(i, i < 2, i % 2 == 0) for i in range(6)]
    report = format_uncertainty_signals_report(rows)
    for signal, _ in [("low_confidence", ""), ("unsupported_article_6", ""),
                      ("hallucinated", "")]:
        assert signal in report


def test_uncertainty_report_handles_empty_results():
    assert "No results to score" in format_uncertainty_signals_report([])


def test_uncertainty_report_is_deterministic():
    rows = [signal_row(i, i < 2, i % 2 == 0) for i in range(6)]
    assert (format_uncertainty_signals_report(rows)
            == format_uncertainty_signals_report(rows))


def test_uncertainty_report_explains_how_to_read_lift():
    rows = [signal_row(i, i < 2, i % 2 == 0) for i in range(6)]
    report = format_uncertainty_signals_report(rows)
    assert "lift > 1" in report


def test_uncertainty_report_survives_an_old_results_file():
    legacy = [{"case_id": 1, "exact_a": True, "flagged": False}]
    report = format_uncertainty_signals_report(legacy)
    assert "predates the split-out signals" in report


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
