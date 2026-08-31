from datasets import load_dataset
from collections import defaultdict
from agents.facts_agent import extract_facts
from agents.precedent_agent import PrecedentAgent, build_precedent_query
from agents.prosecutor_agent import prosecutor_argue, prosecutor_rebut
from agents.defender_agent import defender_argue, defender_respond
from agents.judge_agent import judge_verdict
from evaluation.rag_metrics import check_code_grounding, evaluate_retrieval, score_faithfulness
from config import ARTICLE_CODES
import json
import numpy as np
import os
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

def run_evaluation(
    n_cases: int = 1,
    seed: int = 123,
    judge_faithfulness: bool = False,
    out_path: str = "evaluation/results.json",
):
    """
    Run evaluation on a random sample of cases.

    Retrieval and code-grounding metrics are free (pure set arithmetic on stored labels)
    and always run. judge_faithfulness=True adds one LLM-judge call per case.
    """
    ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
    precedent_agent = PrecedentAgent()

    random.seed(seed)
    case_indices = random.sample(range(len(ds['test'])), n_cases)

    results = []
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
            precedents = precedent_agent.find_precedents(
                paragraphs, query_text=build_precedent_query(facts)
            )
            precedent_analysis = precedent_agent.analyze_precedents(paragraphs, precedents)

            retrieval = evaluate_retrieval(ground_truth, precedents)
            ungrounded_codes = check_code_grounding(precedent_analysis, precedents)
            faithfulness = (
                score_faithfulness(precedent_analysis, precedents)
                if judge_faithfulness else None
            )

            verdict_a = run_variant_a(facts, precedent_analysis)
            verdict_b = run_variant_b(facts, precedent_analysis)

        except Exception as e:
            print(f"  Case {i} skipped due to error: {e}")
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

        results.append({
            "case_id": i,
            "ground_truth": sorted(eval_a["truth"]),
            "predicted_a": sorted(eval_a["predicted"]),
            "predicted_b": sorted(eval_b["predicted"]),
            "exact_a": eval_a["exact_match"], "partial_a": eval_a["partial_match"],
            "exact_b": eval_b["exact_match"], "partial_b": eval_b["partial_match"],
            "flagged": bool(
                verdict_a.get("low_confidence")
                or verdict_a.get("unsupported_article_6")
                or verdict_a.get("filtered_hallucinated_codes")
            ),
            "precision_at_k": retrieval["precision_at_k"],
            "retrieval_hits": retrieval["hits"],
            "ungrounded_codes": ungrounded_codes,
            "faithfulness": faithfulness["faithfulness"] if faithfulness else None,
            "faithfulness_truncated": bool(faithfulness and faithfulness.get("truncated")),
        })

        print(f"  A: exact={'✅' if eval_a['exact_match'] else '❌'} partial={'✅' if eval_a['partial_match'] else '❌'} — {eval_a['predicted']}")
        print(f"  B: exact={'✅' if eval_b['exact_match'] else '❌'} partial={'✅' if eval_b['partial_match'] else '❌'} — {eval_b['predicted']}")
        p_at_k = retrieval['precision_at_k']
        print(f"  RAG: precision@{len(retrieval['hits'])}="
              + (f"{p_at_k:.2f}" if p_at_k is not None else "n/a (no-violation case)")
              + (f", ungrounded codes={ungrounded_codes}" if ungrounded_codes else "")
              + (f", faithfulness={faithfulness['faithfulness']:.2f}"
                 if faithfulness and faithfulness["faithfulness"] is not None else ""))

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

    retrieval_summary = summarize_retrieval(results)
    print("\n=== RETRIEVAL QUALITY ===")
    print(f"Mean precision@k: {retrieval_summary['mean_precision_at_k']:.2f} "
          f"({retrieval_summary['n_hits']} precedents retrieved across "
          f"{retrieval_summary['n_scored_cases']}/{retrieval_summary['n_total_cases']} scored cases; "
          f"no-violation cases excluded)")
    correlation = retrieval_summary["similarity_relevance_correlation"]
    if correlation is None:
        print("Similarity vs. relevance correlation: n/a (needs 2+ precedents with varying relevance)")
    else:
        print(f"Similarity vs. relevance correlation: {correlation:+.2f} "
              "(near 0 means cosine similarity does not predict legal relevance)")

    grounding_summary = summarize_grounding(results)
    print("\n=== PRECEDENT ANALYSIS GROUNDING ===")
    print(f"{grounding_summary['n_cases_with_ungrounded_codes']}/{n} analyses cited an article "
          f"no retrieved precedent supports")
    if grounding_summary["mean_faithfulness"] is not None:
        print(f"Mean faithfulness: {grounding_summary['mean_faithfulness']:.2f} "
              f"(judged on {grounding_summary['n_judged']}/{n} cases)")

    if out_path:
        write_results(out_path, results, per_article_stats, retrieval_summary, grounding_summary)
        print(f"\nResults written to {out_path}")

    return results


def summarize_retrieval(results: list[dict]) -> dict:
    """Aggregate per-case retrieval metrics, including whether similarity tracks relevance."""
    hits = [h for r in results for h in r["retrieval_hits"]]
    similarities = [h["similarity"] for h in hits if h["similarity"] is not None]
    relevances = [float(h["relevant"]) for h in hits if h["similarity"] is not None]

    # Point-biserial correlation. Undefined when either side is constant (e.g. every
    # precedent relevant, or all similarities equal), which numpy reports as nan.
    correlation = None
    if len(similarities) >= 2 and len(set(similarities)) > 1 and len(set(relevances)) > 1:
        correlation = float(np.corrcoef(similarities, relevances)[0, 1])
        if np.isnan(correlation):
            correlation = None

    # No-violation cases have an undefined precision@k (no article to share) and are
    # excluded rather than counted as zero.
    scored = [r["precision_at_k"] for r in results if r["precision_at_k"] is not None]

    return {
        "mean_precision_at_k": sum(scored) / len(scored) if scored else 0.0,
        "n_scored_cases": len(scored),
        "n_total_cases": len(results),
        "n_hits": len(hits),
        "similarity_relevance_correlation": correlation,
    }


def summarize_grounding(results: list[dict]) -> dict:
    """Aggregate the free code-grounding check and, when it ran, the LLM faithfulness judge."""
    judged = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    return {
        "n_cases_with_ungrounded_codes": sum(1 for r in results if r["ungrounded_codes"]),
        "n_judged": len(judged),
        "mean_faithfulness": sum(judged) / len(judged) if judged else None,
    }


def write_results(out_path, results, per_article_stats, retrieval_summary, grounding_summary):
    """Persist the run so metrics survive past the console."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "cases": results,
        # per_article_stats is a nested defaultdict; json.dump needs plain dicts.
        "per_article_stats": {
            variant: {article: dict(stats) for article, stats in articles.items()}
            for variant, articles in per_article_stats.items()
        },
        "retrieval": retrieval_summary,
        "grounding": grounding_summary,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

if __name__ == "__main__":
    run_evaluation(n_cases=15, seed=42)