from datasets import load_dataset
from collections import defaultdict
from datetime import datetime, timezone
from agents.facts_agent import extract_facts
from agents.precedent_agent import PrecedentAgent
from agents.prosecutor_agent import prosecutor_argue, prosecutor_rebut
from agents.defender_agent import defender_argue, defender_respond
from agents.judge_agent import judge_verdict
from config import ARTICLE_CODES, MODEL
from evaluation.calibration import format_calibration_report
from evaluation.results_io import DEFAULT_RESULTS_DIR, save_results
import random


NO_PROSECUTOR_ARGS = "No prosecution arguments provided."
NO_DEFENDER_ARGS = "No defense arguments provided."
NO_REBUTTAL = "No rebuttal provided."
NO_FINAL_RESPONSE = "No final response provided."


def run_variant_a(facts, precedent_analysis):
    """With debate."""
    prosecutor_args = prosecutor_argue(facts, precedent_analysis)
    defender_args = defender_argue(facts, precedent_analysis)
    prosecutor_rebuttal = prosecutor_rebut(facts, precedent_analysis, defender_args)
    defender_final = defender_respond(facts, precedent_analysis, prosecutor_rebuttal)

    return judge_verdict(
        case_facts=facts,
        precedents=precedent_analysis,
        prosecutor_arguments=prosecutor_args,
        defender_arguments=defender_args,
        prosecutor_rebuttal=prosecutor_rebuttal,
        defender_response=defender_final
    )


def run_variant_b(facts, precedent_analysis):
    """Without debate."""
    return judge_verdict(
        case_facts=facts,
        precedents=precedent_analysis,
        prosecutor_arguments=NO_PROSECUTOR_ARGS,
        defender_arguments=NO_DEFENDER_ARGS,
        prosecutor_rebuttal=NO_REBUTTAL,
        defender_response=NO_FINAL_RESPONSE
    )


def evaluate_verdict(verdict: dict, ground_truth_codes: list[str]) -> dict:
    """Returns detailed match info, not just True/False."""
    predicted = set(verdict.get('violated_articles', []))
    truth = set(ground_truth_codes)

    exact_match = predicted == truth
    partial_match = len(predicted & truth) > 0 if truth else len(predicted) == 0

    return {
        "exact_match": exact_match,
        "partial_match": partial_match,
        "predicted": predicted,
        "truth": truth,
    }

def run_evaluation(n_cases: int = 1, seed: int = 123, output_dir=DEFAULT_RESULTS_DIR):
    """Run evaluation on a random sample of cases."""
    ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
    precedent_agent = PrecedentAgent()

    random.seed(seed)
    case_indices = random.sample(range(len(ds['test'])), n_cases)

    results = []
    # Which cases dropped out and why. Without this a systematic failure — say
    # every long case exhausting max_tokens — silently biases the sample and
    # only shows up as a smaller n.
    failures = []
    per_article_stats = {
        "a": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
        "b": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
    }

    for i in case_indices:
        case = ds['test'][i]
        paragraphs = case['text']
        ground_truth = [ARTICLE_CODES[l] for l in case['labels']]

        print(f"\n--- Case {i} (ground truth: {ground_truth}) ---")

        try:
            facts = extract_facts(paragraphs)
            precedents = precedent_agent.find_precedents(paragraphs)
            precedent_analysis = precedent_agent.analyze_precedents(paragraphs, precedents)

            verdict_a = run_variant_a(facts, precedent_analysis)
            verdict_b = run_variant_b(facts, precedent_analysis)

        except Exception as e:
            print(f"  Case {i} skipped due to error: {e}")
            failures.append({"case_id": i, "error": f"{type(e).__name__}: {e}"})
            continue

        eval_a = evaluate_verdict(verdict_a, ground_truth)
        eval_b = evaluate_verdict(verdict_b, ground_truth)

        for variant_key, eval_result in [("a", eval_a), ("b", eval_b)]:
            for article in eval_result["truth"] | eval_result["predicted"]:
                if article in eval_result["predicted"] and article in eval_result["truth"]:
                    per_article_stats[variant_key][article]["tp"] += 1
                elif article in eval_result["predicted"]:
                    per_article_stats[variant_key][article]["fp"] += 1
                elif article in eval_result["truth"]:
                    per_article_stats[variant_key][article]["fn"] += 1

        row = {
            "case_id": i,
            "exact_a": eval_a["exact_match"], "partial_a": eval_a["partial_match"],
            "exact_b": eval_b["exact_match"], "partial_b": eval_b["partial_match"],
            # Ground truth, shared by both variants. An empty label list means
            # the Court found no violation, per the LexGLUE dataset card.
            "truth": sorted(eval_a["truth"]),
            "truth_violation": len(ground_truth) > 0,
        }
        for variant_key, verdict, eval_result in (
            ("a", verdict_a, eval_a), ("b", verdict_b, eval_b)
        ):
            # Sets are not JSON-serialisable, and sorting keeps the file stable
            # across runs so two batches can be diffed.
            row[f"predicted_{variant_key}"] = sorted(eval_result["predicted"])
            row[f"violation_{variant_key}"] = bool(verdict.get("violation"))
            row[f"confidence_{variant_key}"] = verdict.get("confidence_score")
            # Kept apart rather than OR-ed together: "the model reported low
            # confidence" is the model's own signal, while the other two are our
            # code catching something. Only the first belongs in discrimination.
            row[f"low_confidence_{variant_key}"] = bool(verdict.get("low_confidence"))
            row[f"unsupported_article_6_{variant_key}"] = bool(
                verdict.get("unsupported_article_6")
            )
            row[f"hallucinated_{variant_key}"] = list(
                verdict.get("filtered_hallucinated_codes") or []
            )
        # Superseded by the three fields above; kept so the SELF-FLAGGED summary
        # below keeps working until it is rewritten to read them separately.
        row["flagged"] = bool(
            row["low_confidence_a"]
            or row["unsupported_article_6_a"]
            or row["hallucinated_a"]
        )
        results.append(row)

        print(f"  A: exact={'✅' if eval_a['exact_match'] else '❌'} partial={'✅' if eval_a['partial_match'] else '❌'} — {eval_a['predicted']}")
        print(f"  B: exact={'✅' if eval_b['exact_match'] else '❌'} partial={'✅' if eval_b['partial_match'] else '❌'} — {eval_b['predicted']}")

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # The model actually resolved at import, not the one named in the README
        # — they can differ via LEGALMIND_MODEL.
        "model": MODEL,
        # call_claude sets no temperature, so this runs at the API default.
        # Recorded because without it nobody can tell the results are not
        # reproducible run to run.
        "sampling": "sdk default (temperature unset)",
        "n_cases_requested": n_cases,
        "n_cases_completed": len(results),
        "seed": seed,
        "dataset": "coastalcph/lex_glue/ecthr_a",
        "split": "test",
        # Label index -> article code. Pinned so an old results file stays
        # interpretable if this order ever changes.
        "article_codes": list(ARTICLE_CODES),
        "failures": failures,
    }
    results_path = save_results(results, meta, output_dir)
    print(f"\nResults written to {results_path}")

    n = len(results)
    if n == 0:
        print("\nNo cases completed successfully — nothing to summarize.")
        return results

    print("\n=== BATCH SUMMARY ===")
    for variant in ['a', 'b']:
        exact_acc = sum(r[f'exact_{variant}'] for r in results) / n
        partial_acc = sum(r[f'partial_{variant}'] for r in results) / n
        print(f"Variant {variant.upper()}: exact={exact_acc*100:.1f}%, partial={partial_acc*100:.1f}%")

    print("\n=== PER-ARTICLE F1 ===")
    for variant in ['a', 'b']:
        print(f"\nVariant {variant.upper()}:")
        for article, stats in sorted(per_article_stats[variant].items()):
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  Article {article}: precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f} (tp={tp}, fp={fp}, fn={fn})")


    flagged = [r for r in results if r["flagged"]]
    if flagged:
        flagged_correct = sum(r["exact_a"] for r in flagged)
        print(f"\n=== SELF-FLAGGED VERDICTS ===")
        print(f"{len(flagged)}/{n} verdicts self-flagged as uncertain; {flagged_correct}/{len(flagged)} were exact-correct")

    print()
    print(format_calibration_report(results))

    if failures:
        print(f"\n=== FAILURES ===")
        for failure in failures:
            print(f"  Case {failure['case_id']}: {failure['error']}")

    return results

if __name__ == "__main__":
    run_evaluation(n_cases=15, seed=42)