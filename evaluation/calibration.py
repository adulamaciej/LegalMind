"""
Calibration metrics for the Judge agent's confidence_score.

The Judge returns a `confidence_score` alongside every verdict, but the prompt
never defines *what* that confidence refers to. Calibration is only defined
against a specific binary event, so this module scores it against three
explicitly named targets and lets the data show which one (if any) the model
actually tracks:

    violation    — did the model get the binary violation/no-violation call right?
    exact        — did the model predict exactly the right set of articles?
    any_correct  — did the model name at least one correct article?
                   (mirrors `partial_match` in evaluation.py, including its
                   empty-ground-truth special case)

Metrics
-------
brier          Mean squared error between confidence and outcome. Range [0, 1],
               lower is better. Needs no binning, so it is the primary metric
               at small sample sizes.
ece            Expected Calibration Error: bin cases by confidence, then average
               |accuracy - mean confidence| per bin, weighted by bin size.
               Range [0, 1], lower is better. Sensitive to the number of bins
               and unreliable when bins hold only a handful of cases — always
               read it together with the per-bin counts.
overconfidence mean_confidence - accuracy. Signed: positive means the model
               claims more certainty than it earns, negative means it
               understates. The most directly interpretable single number.
accuracy       Fraction of scored cases where the target event was correct.
               This is also the base rate of the event being calibrated
               against, which is why no separate base-rate field is reported.

Calibration answers "when the model says 85%, is it right 85% of the time?".
Discrimination answers the different — and for triage more important — question
"does the model give higher confidence to the cases it gets right?". The two are
orthogonal: a model claiming a constant 70% while being 70% accurate is
perfectly calibrated and useless for triage, while a model that always says 99%
when right and 95% when wrong is badly calibrated but ranks perfectly, so its
threshold can simply be moved.

separation     mean confidence when correct - mean confidence when incorrect,
               in confidence units. Positive means confidence carries signal;
               negative means the model is most confident when it is wrong.
auroc          Probability that a random correct case outscores a random
               incorrect one. 0.5 is a coin flip, 1.0 is perfect ranking, below
               0.5 is anti-correlated. Unlike ECE it is base-rate independent,
               which makes it the right metric for comparing targets.

Both of those say whether a signal exists. error_detection_rate says what the
signal is worth: review the least-confident k% of verdicts and it reports what
share of all errors that catches. Picking the same number of cases at random
would catch k% of errors, so the random baseline is exactly review_fraction and
`lift` = detection_rate / review_fraction. A lift above 1 means triage beats
guessing; below 1 means it is actively harmful, because the reviewed cases are
disproportionately the good ones.

Caveats
-------
ECE is NOT comparable across targets with different base rates, so it must not
be used to infer which target the model tracks — use auroc for that. ECE is also
not comparable across variants whose accuracy differs, because a more accurate
model gets a lower ECE at equal confidence.

At small sample sizes prefer `separation` over `auroc`: with only a handful of
incorrect verdicts, auroc moves in large steps and reads as more precise than it
is. Both are None when every case fell on the same side.

Comparing variant A against B is only paired for Brier, because that is the one
metric with a per-case value. ece, separation, auroc and lift describe a whole
set, so their A/B comparison is two independent scalars with none of the power
that pairing provides — a difference between them is much weaker evidence and
must be labelled as unpaired wherever it is presented.

`no_discrimination` (confidence is constant) does not invalidate brier or ece;
it means ECE collapses to |overconfidence| and carries no extra information,
and that confidence is useless for telling easy cases from hard ones.
"""

from math import ceil, comb, sqrt

TARGETS = ("violation", "exact", "any_correct")

# Fields a result row must carry for a given variant to be scoreable.
_REQUIRED_FIELDS = {
    "violation": ("violation_{v}", "truth_violation"),
    "exact": ("predicted_{v}", "truth"),
    "any_correct": ("predicted_{v}", "truth"),
}


# Heuristic thresholds for naming the target the model's confidence tracks.
# Deliberately explicit rather than buried in the wording of a conclusion: at
# n=15 an auroc gap of a few hundredths is noise, so a winner is only named when
# it clears a floor AND beats the runner-up by a margin.
TRACKED_TARGET_MIN_AUROC = 0.65
TRACKED_TARGET_MIN_MARGIN = 0.10

# Above this accuracy gap between variants, an ECE difference says more about
# accuracy than about calibration.
ACCURACY_GAP_WARNING = 0.10

# The three things judge_agent flags a verdict for. Deliberately kept apart:
# the first is the model's own report about itself, the other two are our code
# catching something the model did. OR-ing them into one boolean hides which of
# the three, if any, actually predicts a wrong verdict.
UNCERTAINTY_SIGNALS = (
    ("low_confidence", "model reported confidence below the threshold"),
    ("unsupported_article_6", "Article 6 named without fair-trial keywords"),
    ("hallucinated", "invented article codes were filtered out"),
)


def normalize_confidence(raw) -> float | None:
    """
    Coerce a model-supplied confidence into a percentage in [0, 100].

    The Judge prompt asks for 0-100 but nothing enforces it, so this tolerates
    numeric strings and fractions (0.85 -> 85.0) and rejects anything it cannot
    interpret. Returns None for unusable values so callers can count exclusions
    instead of silently dropping cases.
    """
    if isinstance(raw, bool):  # bool is an int subclass; not a confidence
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    if 0.0 <= value <= 1.0:
        value *= 100
    if not 0.0 <= value <= 100.0:
        return None
    return value


def _outcome(row: dict, variant: str, target: str) -> bool:
    """Was the verdict correct, for this target?"""
    if target == "violation":
        return bool(row[f"violation_{variant}"]) == bool(row["truth_violation"])
    if target == "exact":
        return set(row[f"predicted_{variant}"]) == set(row["truth"])
    if target == "any_correct":
        predicted, truth = set(row[f"predicted_{variant}"]), set(row["truth"])
        return len(predicted & truth) > 0 if truth else len(predicted) == 0
    raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")


def _check_schema(results: list[dict], variant: str, target: str) -> None:
    """
    Fail loudly on result files written before calibration fields existed.

    Every row is checked, not just the first — a partially-migrated file would
    otherwise raise a bare KeyError halfway through scoring.
    """
    required = [f.format(v=variant) for f in _REQUIRED_FIELDS[target]]
    required.append(f"confidence_{variant}")
    for index, row in enumerate(results):
        missing = [f for f in required if f not in row]
        if missing:
            raise ValueError(
                f"result row {index} is missing {missing} — this results file "
                f"predates calibration support; re-run the evaluation to "
                f"regenerate it"
            )


def _bin_index(confidence: float, n_bins: int) -> int:
    """Map a confidence in [0, 1] to an equal-width bin, 1.0 landing in the last."""
    return min(int(confidence * n_bins), n_bins - 1)


def _scored_cases(results: list[dict], variant: str, target: str):
    """
    Extract (confidence in [0, 1], case_id, outcome) triples and an exclusion count.

    Shared by every metric so they cannot drift apart on which cases they
    consider scoreable.
    """
    scored = []
    n_excluded = 0
    for row in results:
        confidence = normalize_confidence(row.get(f"confidence_{variant}"))
        if confidence is None:
            n_excluded += 1
            continue
        scored.append(
            (confidence / 100, row.get("case_id"), _outcome(row, variant, target))
        )
    return scored, n_excluded


def _paired_cases(results: list[dict], target: str):
    """
    Cases scoreable in BOTH variants, as
    (case_id, confidence_a, outcome_a, confidence_b, outcome_b).

    A separate extraction from _scored_cases because pairing needs both
    variants present on the same row, but it reuses the same normalize and
    outcome primitives so the three call sites agree on what counts.
    """
    paired = []
    for row in results:
        confidence_a = normalize_confidence(row.get("confidence_a"))
        confidence_b = normalize_confidence(row.get("confidence_b"))
        if confidence_a is None or confidence_b is None:
            continue
        paired.append(
            (
                row.get("case_id"),
                confidence_a / 100,
                _outcome(row, "a", target),
                confidence_b / 100,
                _outcome(row, "b", target),
            )
        )
    return paired


def _midranks(values: list[float]) -> list[float]:
    """
    1-based ranks with ties sharing their average rank.

    Tie correction is load-bearing here: models favour round confidences like
    85 and 90, so ties are the common case rather than an edge case, and
    ranking them arbitrarily would bias AUROC.
    """
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _auroc(scored: list[tuple[float, object, bool]]) -> float | None:
    """
    Probability that a random correct case outscores a random incorrect one.

    Mann-Whitney U over tie-corrected ranks. Returns None when either group is
    empty, since there is nothing to rank against.
    """
    correct_ranks = []
    n_correct = n_incorrect = 0
    ranks = _midranks([confidence for confidence, _, _ in scored])
    for rank, (_, _, outcome) in zip(ranks, scored):
        if outcome:
            correct_ranks.append(rank)
            n_correct += 1
        else:
            n_incorrect += 1
    if n_correct == 0 or n_incorrect == 0:
        return None
    u = sum(correct_ranks) - n_correct * (n_correct + 1) / 2
    return u / (n_correct * n_incorrect)


def compute_calibration(
    results: list[dict],
    variant: str = "a",
    target: str = "violation",
    n_bins: int = 5,
) -> dict:
    """
    Calibration of confidence_{variant} against the given target.

    Always returns the same keys, whatever the data looks like — degenerate
    inputs are reported via flags rather than a different return shape, so
    callers never have to branch before reading a metric.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    _check_schema(results, variant, target)

    scored, n_excluded = _scored_cases(results, variant, target)
    n = len(scored)
    # Stable, sorted bin list — emitted even when empty, so two runs diff cleanly.
    bins = [
        {
            "lo": i / n_bins,
            "hi": (i + 1) / n_bins,
            "n": 0,
            "mean_confidence": None,
            "accuracy": None,
        }
        for i in range(n_bins)
    ]

    if n == 0:
        return {
            "variant": variant,
            "target": target,
            "n_scored": 0,
            "n_excluded": n_excluded,
            "brier": None,
            "ece": None,
            "mean_confidence": None,
            "accuracy": None,
            "overconfidence": None,
            "no_discrimination": True,
            "uninformative_target": True,
            "distinct_confidence_values": [],
            "bins": bins,
        }

    confidences = [c for c, _, _ in scored]
    outcomes = [o for _, _, o in scored]

    mean_confidence = sum(confidences) / n
    accuracy = sum(outcomes) / n
    brier = sum((c - o) ** 2 for c, _, o in scored) / n

    # Accumulate per bin, then derive ECE from bin means (not bin midpoints —
    # midpoints bias the estimate when a bin's values are skewed, which they
    # are here because models favour round numbers like 85 and 90).
    totals = [{"n": 0, "conf_sum": 0.0, "correct": 0} for _ in range(n_bins)]
    for confidence, _, outcome in scored:
        acc = totals[_bin_index(confidence, n_bins)]
        acc["n"] += 1
        acc["conf_sum"] += confidence
        acc["correct"] += int(outcome)

    ece = 0.0
    for i, acc in enumerate(totals):
        if acc["n"] == 0:
            continue
        bin_confidence = acc["conf_sum"] / acc["n"]
        bin_accuracy = acc["correct"] / acc["n"]
        bins[i]["n"] = acc["n"]
        bins[i]["mean_confidence"] = bin_confidence
        bins[i]["accuracy"] = bin_accuracy
        ece += (acc["n"] / n) * abs(bin_accuracy - bin_confidence)

    return {
        "variant": variant,
        "target": target,
        "n_scored": n,
        "n_excluded": n_excluded,
        "brier": brier,
        "ece": ece,
        "mean_confidence": mean_confidence,
        "accuracy": accuracy,
        "overconfidence": mean_confidence - accuracy,
        "no_discrimination": len(set(confidences)) < 2,
        "uninformative_target": len(set(outcomes)) < 2,
        "distinct_confidence_values": sorted(round(c * 100, 4) for c in set(confidences)),
        "bins": bins,
    }


def sign_test(n_wins: int, n_decisive: int) -> float | None:
    """
    Exact two-sided p-value for "A wins as often as B".

    Under the null hypothesis that the debate makes no difference, each case is
    a coin flip, so the win count is Binomial(n_decisive, 0.5). Exact rather
    than bootstrapped: no seed to record and the output is reproducible to the
    byte.

    Deliberately ignores the size of each win — one large margin counts the same
    as one narrow one. Read it alongside mean_delta, which carries magnitude.

    Returns None when no case was decisive, since there is nothing to test.
    """
    if n_decisive <= 0:
        return None
    if not 0 <= n_wins <= n_decisive:
        raise ValueError(
            f"n_wins must be in [0, {n_decisive}], got {n_wins}"
        )
    extreme = max(n_wins, n_decisive - n_wins)
    tail = sum(comb(n_decisive, i) for i in range(extreme, n_decisive + 1))
    return min(1.0, 2 * tail / 2 ** n_decisive)


def compare_variants_paired(results: list[dict], target: str = "violation") -> dict:
    """
    Per-case Brier comparison of variant A (with debate) against B (without).

    Both variants run on the same cases from the same extracted facts and
    precedents, so the comparison is paired — which at n=15 is the only source
    of statistical power available.

    Only Brier can be paired. separation, auroc, ece and lift are properties of
    a whole set with no per-case value, so their A/B comparison is necessarily
    unpaired and belongs in the report rather than here.

    Note that mean_delta equals brier_a - brier_b exactly; the pairing buys the
    distribution (win counts, spread, sign test), not a new point estimate.

    Lower Brier is better, so a negative delta favours A.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")

    _check_schema(results, "a", target)
    _check_schema(results, "b", target)

    # Reported so the reader can see why n_paired may be smaller than either.
    scored_a, _ = _scored_cases(results, "a", target)
    scored_b, _ = _scored_cases(results, "b", target)

    paired = _paired_cases(results, target)
    n_paired = len(paired)

    per_case = []
    for case_id, confidence_a, outcome_a, confidence_b, outcome_b in paired:
        brier_a = (confidence_a - outcome_a) ** 2
        brier_b = (confidence_b - outcome_b) ** 2
        per_case.append(
            {
                "case_id": case_id,
                "brier_a": brier_a,
                "brier_b": brier_b,
                "delta": brier_a - brier_b,
            }
        )

    if n_paired == 0:
        return {
            "target": target,
            "n_paired": 0,
            "n_scored_a": len(scored_a),
            "n_scored_b": len(scored_b),
            "brier_a": None,
            "brier_b": None,
            "mean_delta": None,
            "sd_delta": None,
            "min_delta": None,
            "max_delta": None,
            "favors_a": 0,
            "favors_b": 0,
            "ties": 0,
            "n_decisive": 0,
            "sign_test_p": None,
            "undefined_reason": "no cases scoreable in both variants",
            "per_case": [],
        }

    deltas = [case["delta"] for case in per_case]
    mean_delta = sum(deltas) / n_paired
    if n_paired > 1:
        variance = sum((d - mean_delta) ** 2 for d in deltas) / (n_paired - 1)
        sd_delta = sqrt(variance)
    else:
        sd_delta = None

    favors_a = sum(1 for d in deltas if d < 0)
    favors_b = sum(1 for d in deltas if d > 0)
    ties = sum(1 for d in deltas if d == 0)
    n_decisive = favors_a + favors_b

    return {
        "target": target,
        "n_paired": n_paired,
        "n_scored_a": len(scored_a),
        "n_scored_b": len(scored_b),
        # Computed on the paired subset so every number here reconciles.
        "brier_a": sum(case["brier_a"] for case in per_case) / n_paired,
        "brier_b": sum(case["brier_b"] for case in per_case) / n_paired,
        "mean_delta": mean_delta,
        "sd_delta": sd_delta,
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "favors_a": favors_a,
        "favors_b": favors_b,
        "ties": ties,
        "n_decisive": n_decisive,
        # None when every paired case tied; the Brier comparison is still valid.
        "sign_test_p": sign_test(favors_a, n_decisive),
        "undefined_reason": None,
        "per_case": per_case,
    }


def error_detection_rate(
    results: list[dict],
    variant: str = "a",
    target: str = "violation",
    review_fractions: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> dict:
    """
    If a human reviews the least-confident k% of verdicts, what share of the
    errors does that catch?

    This is the decision-relevant form of discrimination, and the direct test of
    whether judge_agent's CONFIDENCE_THRESHOLD earns its place: sorting by
    confidence is only worth doing if it beats reviewing the same number of
    cases at random.

    Reviewing k% of cases at random catches k% of the errors in expectation, so
    the random baseline is review_fraction itself and `lift` is the ratio
    against it. Budgets round up, because half a case cannot be reviewed and
    rounding down would silently produce an empty budget at small n.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")
    for fraction in review_fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"review_fractions must be in (0, 1], got {fraction}"
            )

    _check_schema(results, variant, target)

    scored, n_excluded = _scored_cases(results, variant, target)
    # Sort by confidence, then case_id, so a tie group is ordered the same way on
    # every run rather than depending on input order. str() on the tiebreaker so
    # a row without a case_id sorts deterministically instead of raising on a
    # None-vs-int comparison; only determinism matters here, not the ordering
    # being meaningful.
    scored.sort(key=lambda item: (item[0], str(item[1])))

    n = len(scored)
    n_errors = sum(1 for _, _, outcome in scored if not outcome)

    budgets = []
    for fraction in sorted(review_fractions):
        n_reviewed = min(n, ceil(fraction * n)) if n else 0
        reviewed = scored[:n_reviewed]
        caught = sum(1 for _, _, outcome in reviewed if not outcome)
        detection_rate = caught / n_errors if n_errors else None
        # A budget that cuts through a group of equal confidences includes an
        # arbitrary subset of it, so the number is not reproducible under a
        # reshuffle of the input. Flag it rather than presenting it as stable.
        boundary_ties = (
            0 < n_reviewed < n and scored[n_reviewed - 1][0] == scored[n_reviewed][0]
        )
        budgets.append(
            {
                "review_fraction": fraction,
                "n_reviewed": n_reviewed,
                "errors_caught": caught,
                "detection_rate": detection_rate,
                "random_baseline": fraction,
                "lift": detection_rate / fraction if detection_rate is not None else None,
                "boundary_ties": boundary_ties,
            }
        )

    return {
        "variant": variant,
        "target": target,
        "n_scored": n,
        "n_excluded": n_excluded,
        "n_errors": n_errors,
        "undefined_reason": None if n_errors else (
            "no scoreable cases" if n == 0 else "no errors to detect"
        ),
        "budgets": budgets,
    }


def compute_discrimination(
    results: list[dict],
    variant: str = "a",
    target: str = "violation",
) -> dict:
    """
    Does confidence_{variant} separate correct verdicts from incorrect ones?

    Discrimination is what a triage mechanism actually needs — `low_confidence`
    in judge_agent.py is only useful if the flagged verdicts really are the
    wrong ones. A perfectly calibrated model with constant confidence has zero
    discrimination and is useless for triage; a badly calibrated model that
    ranks correctly is useful, because the threshold can just be moved.

    Both metrics are undefined when every case landed on the same side (nothing
    to separate), in which case they are None and `undefined_reason` says why.
    The return shape never changes.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")

    _check_schema(results, variant, target)

    scored, n_excluded = _scored_cases(results, variant, target)
    correct = [confidence for confidence, _, outcome in scored if outcome]
    incorrect = [confidence for confidence, _, outcome in scored if not outcome]
    n_correct, n_incorrect = len(correct), len(incorrect)

    if n_correct == 0 or n_incorrect == 0:
        if not scored:
            reason = "no scoreable cases"
        elif n_incorrect == 0:
            reason = "every scored case was correct"
        else:
            reason = "every scored case was incorrect"
        return {
            "variant": variant,
            "target": target,
            "n_scored": len(scored),
            "n_excluded": n_excluded,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "mean_confidence_when_correct": sum(correct) / n_correct if n_correct else None,
            "mean_confidence_when_incorrect": (
                sum(incorrect) / n_incorrect if n_incorrect else None
            ),
            "separation": None,
            "auroc": None,
            "undefined_reason": reason,
        }

    mean_correct = sum(correct) / n_correct
    mean_incorrect = sum(incorrect) / n_incorrect

    return {
        "variant": variant,
        "target": target,
        "n_scored": len(scored),
        "n_excluded": n_excluded,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "mean_confidence_when_correct": mean_correct,
        "mean_confidence_when_incorrect": mean_incorrect,
        # Positive means confidence carries signal; negative means the model is
        # most confident exactly when it is wrong, which is worse than useless.
        "separation": mean_correct - mean_incorrect,
        "auroc": _auroc(scored),
        "undefined_reason": None,
    }


def signal_error_lift(
    results: list[dict],
    signal: str,
    variant: str = "a",
    target: str = "exact",
) -> dict:
    """
    Does an uncertainty flag actually mark verdicts that are more often wrong?

    A flag earns its place only if the verdicts it fires on are wrong more often
    than average. `lift` is the error rate among flagged verdicts over the error
    rate across all of them: above 1 the flag points at trouble, at 1 it is
    uninformative, below 1 it points away from it.

    `coverage` is the other half — a flag that fires once and is right about it
    still leaves the other errors unflagged.

    Also reports mean confidence on each side, which answers a question the
    combined flag could not: does the model sound more or less certain when it
    invents an article code?
    """
    field = f"{signal}_{variant}"
    missing = [i for i, row in enumerate(results) if field not in row]
    if missing:
        return {
            "signal": signal,
            "variant": variant,
            "target": target,
            "n_total": len(results),
            "n_fired": None,
            "error_rate_when_fired": None,
            "error_rate_overall": None,
            "lift": None,
            "coverage": None,
            "mean_confidence_when_fired": None,
            "mean_confidence_when_quiet": None,
            "undefined_reason": (
                f"'{field}' is absent from {len(missing)} of {len(results)} rows "
                f"— this results file predates the split-out signals"
            ),
        }

    _check_schema(results, variant, target)

    fired, quiet = [], []
    for row in results:
        record = (
            bool(row[field]),
            _outcome(row, variant, target),
            normalize_confidence(row.get(f"confidence_{variant}")),
        )
        (fired if record[0] else quiet).append(record)

    n_total = len(results)
    n_fired = len(fired)
    n_errors_total = sum(1 for _, ok, _ in fired + quiet if not ok)
    n_errors_fired = sum(1 for _, ok, _ in fired if not ok)

    error_rate_overall = n_errors_total / n_total if n_total else None
    error_rate_when_fired = n_errors_fired / n_fired if n_fired else None
    lift = (
        error_rate_when_fired / error_rate_overall
        if error_rate_when_fired is not None and error_rate_overall
        else None
    )

    def _mean_confidence(group):
        values = [c for _, _, c in group if c is not None]
        return sum(values) / len(values) / 100 if values else None

    if n_fired == 0:
        reason = "the signal never fired"
    elif not n_errors_total:
        reason = "no errors to point at"
    else:
        reason = None

    return {
        "signal": signal,
        "variant": variant,
        "target": target,
        "n_total": n_total,
        "n_fired": n_fired,
        "error_rate_when_fired": error_rate_when_fired,
        "error_rate_overall": error_rate_overall,
        "lift": lift,
        # Share of all errors this flag caught.
        "coverage": n_errors_fired / n_errors_total if n_errors_total else None,
        "mean_confidence_when_fired": _mean_confidence(fired),
        "mean_confidence_when_quiet": _mean_confidence(quiet),
        "undefined_reason": reason,
    }


def _fmt(value, places: int = 3, signed: bool = False) -> str:
    """Render a metric, or a dash when it is undefined. Never invents a number."""
    if value is None:
        return "—"
    return f"{value:+.{places}f}" if signed else f"{value:.{places}f}"


def _tracked_target(discriminations: dict) -> tuple[str | None, str, int]:
    """
    Which target does confidence track most closely?

    Decided on auroc, never ece: auroc is base-rate independent, and the targets
    have different base rates, so an ece ranking would mostly reflect which
    target has the most lopsided outcome distribution.

    Returns (target, reason, n_rankable). A target that clears the floor while
    being the only rankable one has not beaten anything — the other targets were
    unmeasurable, not worse — so n_rankable lets the caller say that instead of
    implying a comparison that never happened.
    """
    ranked = sorted(
        ((target, d["auroc"]) for target, d in discriminations.items()
         if d["auroc"] is not None),
        key=lambda item: item[1],
        reverse=True,
    )
    n_rankable = len(ranked)
    if not ranked:
        return None, "no target had both correct and incorrect verdicts to rank", 0
    best_target, best_auroc = ranked[0]
    if best_auroc < TRACKED_TARGET_MIN_AUROC:
        return None, (
            f"best auroc {best_auroc:.3f} is below the "
            f"{TRACKED_TARGET_MIN_AUROC} floor — no usable ranking signal"
        ), n_rankable
    if n_rankable > 1:
        runner_up_target, runner_up_auroc = ranked[1]
        margin = best_auroc - runner_up_auroc
        if margin < TRACKED_TARGET_MIN_MARGIN:
            return None, (
                f"top two targets are within {margin:.3f} auroc, below the "
                f"{TRACKED_TARGET_MIN_MARGIN} margin — too close to call"
            ), n_rankable
        return best_target, (
            f"auroc {best_auroc:.3f}, {margin:.3f} clear of '{runner_up_target}'"
        ), n_rankable
    return best_target, f"auroc {best_auroc:.3f}", n_rankable


def format_calibration_report(results: list[dict], n_bins: int = 5) -> str:
    """
    Render every metric for both variants across all three targets.

    Output is deterministic: the same results produce the same bytes, so two
    runs can be diffed. Nothing time-varying or dict-order-dependent goes in.

    Undefined metrics collapse to a one-line reason rather than an empty table,
    and the warnings that invalidate a reading are printed above the numbers
    they affect rather than as a footnote.
    """
    lines = ["=== CONFIDENCE CALIBRATION ==="]

    if not results:
        lines.append("No results to score.")
        return "\n".join(lines)

    discriminations = {}

    for target in TARGETS:
        cal_a = compute_calibration(results, "a", target, n_bins)
        cal_b = compute_calibration(results, "b", target, n_bins)
        disc_a = compute_discrimination(results, "a", target)
        disc_b = compute_discrimination(results, "b", target)
        detect_a = error_detection_rate(results, "a", target)
        detect_b = error_detection_rate(results, "b", target)
        paired = compare_variants_paired(results, target)
        discriminations[target] = disc_a

        lines.append("")
        lines.append(f"--- target: {target} ---")
        lines.append(
            f"scored: A={cal_a['n_scored']} B={cal_b['n_scored']} "
            f"(excluded: A={cal_a['n_excluded']} B={cal_b['n_excluded']})"
        )

        # Warnings first — each one changes how the numbers below may be read.
        accuracy_gap = (
            abs(cal_a["accuracy"] - cal_b["accuracy"])
            if cal_a["accuracy"] is not None and cal_b["accuracy"] is not None
            else None
        )
        if accuracy_gap is not None and accuracy_gap > ACCURACY_GAP_WARNING:
            lines.append(
                f"  ! accuracy differs by {accuracy_gap:.3f} — the ece gap "
                f"reflects accuracy, not calibration; read overconfidence instead"
            )
        for label, cal in (("A", cal_a), ("B", cal_b)):
            if cal["no_discrimination"] and cal["n_scored"]:
                lines.append(
                    f"  ! {label}: confidence is constant at "
                    f"{cal['distinct_confidence_values']} — ece collapses to "
                    f"|overconfidence| and carries no extra information"
                )
            if cal["uninformative_target"] and cal["n_scored"]:
                lines.append(
                    f"  ! {label}: every scored case had the same outcome — "
                    f"calibration against this target is uninformative"
                )

        lines.append("  calibration (unpaired)")
        lines.append(
            f"    {'':14}{'A':>10}{'B':>10}"
        )
        for label, key in (
            ("accuracy", "accuracy"),
            ("mean conf.", "mean_confidence"),
            ("overconfid.", "overconfidence"),
            ("brier", "brier"),
            ("ece", "ece"),
        ):
            signed = key == "overconfidence"
            lines.append(
                f"    {label:14}{_fmt(cal_a[key], signed=signed):>10}"
                f"{_fmt(cal_b[key], signed=signed):>10}"
            )

        lines.append("  discrimination (unpaired)")
        if disc_a["undefined_reason"] and disc_b["undefined_reason"]:
            lines.append(f"    undefined — {disc_a['undefined_reason']}")
        else:
            lines.append(f"    {'':14}{'A':>10}{'B':>10}")
            lines.append(
                f"    {'separation':14}{_fmt(disc_a['separation'], signed=True):>10}"
                f"{_fmt(disc_b['separation'], signed=True):>10}"
            )
            lines.append(
                f"    {'auroc':14}{_fmt(disc_a['auroc']):>10}"
                f"{_fmt(disc_b['auroc']):>10}"
            )

        lines.append("  error detection at review budget (unpaired)")
        if detect_a["undefined_reason"] and detect_b["undefined_reason"]:
            lines.append(f"    undefined — {detect_a['undefined_reason']}")
        else:
            lines.append(
                f"    {'budget':14}{'A lift':>10}{'B lift':>10}   (1.0 = no better than random)"
            )
            for budget_a, budget_b in zip(detect_a["budgets"], detect_b["budgets"]):
                flag = " !ties" if budget_a["boundary_ties"] or budget_b["boundary_ties"] else ""
                noun = "case" if budget_a["n_reviewed"] == 1 else "cases"
                budget_label = f"{budget_a['review_fraction']:.0%} ({budget_a['n_reviewed']} {noun})"
                lines.append(
                    f"    {budget_label:14}{_fmt(budget_a['lift'], 2):>10}"
                    f"{_fmt(budget_b['lift'], 2):>10}{flag}"
                )

        lines.append("  A vs B (paired — same cases, the only powered comparison)")
        if paired["undefined_reason"]:
            lines.append(f"    undefined — {paired['undefined_reason']}")
        else:
            lines.append(
                f"    paired cases: {paired['n_paired']} "
                f"(A scored {paired['n_scored_a']}, B scored {paired['n_scored_b']})"
            )
            lines.append(
                f"    brier: A={_fmt(paired['brier_a'])} B={_fmt(paired['brier_b'])} "
                f"delta={_fmt(paired['mean_delta'], signed=True)} "
                f"(sd={_fmt(paired['sd_delta'])})"
            )
            if paired["n_decisive"]:
                lines.append(
                    f"    A better in {paired['favors_a']} of "
                    f"{paired['n_decisive']} decisive cases "
                    f"({paired['ties']} tied), sign test p="
                    f"{_fmt(paired['sign_test_p'])}"
                )
            else:
                lines.append(
                    f"    every paired case tied — sign test undefined"
                )

    lines.append("")
    lines.append("--- what does confidence track? ---")
    tracked, reason, n_rankable = _tracked_target(discriminations)
    for target in TARGETS:
        lines.append(f"  {target:12} auroc {_fmt(discriminations[target]['auroc'])}")
    if tracked and n_rankable > 1:
        lines.append(f"  -> confidence tracks '{tracked}' most closely ({reason})")
    elif tracked:
        # Sole rankable target: it cleared the floor but beat nothing, because
        # the others had no incorrect verdicts to rank against.
        lines.append(
            f"  -> '{tracked}' was the only rankable target ({reason}); the "
            f"others had no incorrect verdicts, so no comparison was possible"
        )
    else:
        lines.append(f"  -> no target clearly tracked: {reason}")
    lines.append(
        f"  (decided on auroc, never ece: the targets have different base rates)"
    )

    return "\n".join(lines)


def format_uncertainty_signals_report(
    results: list[dict],
    variant: str = "a",
    target: str = "exact",
) -> str:
    """
    Score each of judge_agent's three uncertainty flags on its own.

    The previous summary OR-ed them together and reported how many flagged
    verdicts were correct, which could not say which flag carried the signal —
    or whether any of them did.
    """
    lines = [f"=== UNCERTAINTY SIGNALS (variant {variant.upper()}, target: {target}) ==="]

    if not results:
        lines.append("No results to score.")
        return "\n".join(lines)

    lines.append("lift > 1 means the flag marks verdicts that are wrong more often than average")

    for signal, description in UNCERTAINTY_SIGNALS:
        stats = signal_error_lift(results, signal, variant, target)
        lines.append("")
        lines.append(f"  {signal} — {description}")
        if stats["undefined_reason"]:
            lines.append(
                f"    fired on {_fmt(stats['n_fired'], 0)} of {stats['n_total']} "
                f"— undefined: {stats['undefined_reason']}"
            )
            continue
        lines.append(
            f"    fired on {stats['n_fired']} of {stats['n_total']} verdicts"
        )
        lines.append(
            f"    error rate: {_fmt(stats['error_rate_when_fired'])} when fired "
            f"vs {_fmt(stats['error_rate_overall'])} overall "
            f"-> lift {_fmt(stats['lift'], 2)}"
        )
        lines.append(
            f"    caught {_fmt(stats['coverage'])} of all errors"
        )
        lines.append(
            f"    mean confidence: {_fmt(stats['mean_confidence_when_fired'])} when "
            f"fired vs {_fmt(stats['mean_confidence_when_quiet'])} when quiet"
        )

    return "\n".join(lines)
