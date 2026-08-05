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

Caveats
-------
ECE is NOT comparable across targets with different base rates, so it must not
be used to infer which target the model tracks — use auroc for that. ECE is also
not comparable across variants whose accuracy differs, because a more accurate
model gets a lower ECE at equal confidence.

At small sample sizes prefer `separation` over `auroc`: with only a handful of
incorrect verdicts, auroc moves in large steps and reads as more precise than it
is. Both are None when every case fell on the same side.

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


def _score_pairs(results: list[dict], variant: str, target: str):
    """
    Extract (confidence in [0, 1], outcome) pairs plus an exclusion count.

    Shared by every metric so they cannot drift apart on which cases they
    consider scoreable.
    """
    pairs = []
    n_excluded = 0
    for row in results:
        confidence = normalize_confidence(row.get(f"confidence_{variant}"))
        if confidence is None:
            n_excluded += 1
            continue
        pairs.append((confidence / 100, _outcome(row, variant, target)))
    return pairs, n_excluded


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


def _auroc(pairs: list[tuple[float, bool]]) -> float | None:
    """
    Probability that a random correct case outscores a random incorrect one.

    Mann-Whitney U over tie-corrected ranks. Returns None when either group is
    empty, since there is nothing to rank against.
    """
    correct_ranks = []
    n_correct = n_incorrect = 0
    ranks = _midranks([confidence for confidence, _ in pairs])
    for rank, (_, outcome) in zip(ranks, pairs):
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

    pairs, n_excluded = _score_pairs(results, variant, target)
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

    pairs, n_excluded = _score_pairs(results, variant, target)
    correct = [confidence for confidence, outcome in pairs if outcome]
    incorrect = [confidence for confidence, outcome in pairs if not outcome]
    n_correct, n_incorrect = len(correct), len(incorrect)

    if n_correct == 0 or n_incorrect == 0:
        if not pairs:
            reason = "no scoreable cases"
        elif n_incorrect == 0:
            reason = "every scored case was correct"
        else:
            reason = "every scored case was incorrect"
        return {
            "variant": variant,
            "target": target,
            "n_scored": len(pairs),
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
        "n_scored": len(pairs),
        "n_excluded": n_excluded,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "mean_confidence_when_correct": mean_correct,
        "mean_confidence_when_incorrect": mean_incorrect,
        # Positive means confidence carries signal; negative means the model is
        # most confident exactly when it is wrong, which is worse than useless.
        "separation": mean_correct - mean_incorrect,
        "auroc": _auroc(pairs),
        "undefined_reason": None,
    }
