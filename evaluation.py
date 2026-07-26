from datasets import load_dataset
from agents.facts_agent import extract_facts
from agents.precedent_agent import PrecedentAgent
from agents.prosecutor_agent import prosecutor_argue, prosecutor_rebut
from agents.defender_agent import defender_argue, defender_respond
from agents.judge_agent import judge_verdict
from config import ARTICLE_CODES


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


def check_correct(verdict, ground_truth_codes):
    predicted = set(verdict.get('violated_articles', []))
    truth = set(ground_truth_codes)
    return predicted == truth


def run_evaluation(start_idx: int = 0, n_cases: int = 3):
    """Run evaluation on a batch of cases, starting at start_idx."""
    ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
    precedent_agent = PrecedentAgent()

    results = []

    for i in range(start_idx, start_idx + n_cases):
        case = ds['test'][i]
        paragraphs = case['text']
        ground_truth = [ARTICLE_CODES[l] for l in case['labels']]

        print(f"\n--- Case {i} (ground truth: {ground_truth}) ---")

        facts = extract_facts(paragraphs)
        precedents = precedent_agent.find_precedents(paragraphs)
        precedent_analysis = precedent_agent.analyze_precedents(paragraphs, precedents)

        verdict_a = run_variant_a(facts, precedent_analysis)
        verdict_b = run_variant_b(facts, precedent_analysis)

        correct_a = check_correct(verdict_a, ground_truth)
        correct_b = check_correct(verdict_b, ground_truth)

        results.append({"case_id": i, "correct_a": correct_a, "correct_b": correct_b})

        print(f"  A (with debate):    {'✅' if correct_a else '❌'} — {verdict_a.get('violated_articles')}")
        print(f"  B (no debate):      {'✅' if correct_b else '❌'} — {verdict_b.get('violated_articles')}")

    print("\n=== BATCH SUMMARY ===")
    n = len(results)
    acc_a = sum(r['correct_a'] for r in results) / n
    acc_b = sum(r['correct_b'] for r in results) / n
    print(f"Variant A (with debate): {acc_a*100:.1f}% ({sum(r['correct_a'] for r in results)}/{n})")
    print(f"Variant B (no debate):   {acc_b*100:.1f}% ({sum(r['correct_b'] for r in results)}/{n})")

    return results


if __name__ == "__main__":
    run_evaluation(start_idx=0, n_cases=3)