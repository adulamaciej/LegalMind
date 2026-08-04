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

Caveats
-------
ECE is NOT comparable across targets with different base rates, so it must not
be used to infer which target the model tracks — use the discrimination metrics
for that. ECE is also not comparable across variants whose accuracy differs,
because a more accurate model gets a lower ECE at equal confidence.

`no_discrimination` (confidence is constant) does not invalidate brier or ece;
it means ECE collapses to |overconfidence| and carries no extra information,
and that confidence is useless for telling easy cases from hard ones.
"""

TARGETS = ("violation", "exact", "any_correct")

# Fields a result row must carry for a given variant to be scoreable.
_REQUIRED_FIELDS = {
    "violation": ("violation_{v}", "truth_violation"),
    "exact": ("predicted_{v}", "truth"),
    "any_correct": ("predicted_{v}", "truth"),
}


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

    pairs = []          # (confidence in [0, 1], outcome as bool)
    n_excluded = 0
    for row in results:
        confidence = normalize_confidence(row.get(f"confidence_{variant}"))
        if confidence is None:
            n_excluded += 1
            continue
        pairs.append((confidence / 100, _outcome(row, variant, target)))

    n = len(pairs)
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

    confidences = [c for c, _ in pairs]
    outcomes = [o for _, o in pairs]

    mean_confidence = sum(confidences) / n
    accuracy = sum(outcomes) / n
    brier = sum((c - o) ** 2 for c, o in pairs) / n

    # Accumulate per bin, then derive ECE from bin means (not bin midpoints —
    # midpoints bias the estimate when a bin's values are skewed, which they
    # are here because models favour round numbers like 85 and 90).
    totals = [{"n": 0, "conf_sum": 0.0, "correct": 0} for _ in range(n_bins)]
    for confidence, outcome in pairs:
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
