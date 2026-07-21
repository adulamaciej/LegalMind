from agents.facts_agent import extract_facts
from agents.precedent_agent import PrecedentAgent
from agents.prosecutor_agent import prosecutor_argue, prosecutor_rebut
from agents.defender_agent import defender_argue, defender_respond
from agents.judge_agent import judge_verdict, format_verdict


def run_pipeline(case_paragraphs: list[str]) -> dict:


    """
    Main LegalMind pipeline.
    Input: list of case fact paragraphs
    Output: full analysis result with verdict
    
    """
    
    print("\n⚖️  LegalMind — ECHR Analysis System")
    print("=" * 50)
    

    # STEP 1: Facts Extraction Agent
    print("\n📋 [1/5] Facts Agent — extracting facts...")
    facts = extract_facts(case_paragraphs)
    print(f"✅ Facts extracted: {facts['summary']}")

    
    # STEP 2: Precedent Agent (RAG)
    print("\n🔍 [2/5] Precedent Agent — searching for precedents...")
    precedent_agent = PrecedentAgent()
    precedents = precedent_agent.find_precedents(case_paragraphs)
    precedent_analysis = precedent_agent.analyze_precedents(case_paragraphs, precedents)
    print(f"✅ Found {len(precedents)} precedents")
    

    # STEP 3: Adversarial Debate
    print("\n⚔️  [3/5] Debate — Prosecutor vs Defender...")
    

    # Prosecutor builds arguments
    prosecutor_args = prosecutor_argue(facts, precedent_analysis)
    print("✅ Prosecutor arguments ready")
    

    # Defender builds arguments independently
    defender_args = defender_argue(facts, precedent_analysis)
    print("✅ Defender arguments ready")
    

    # Prosecutor rebuttal
    prosecutor_rebuttal = prosecutor_rebut(facts, precedent_analysis, defender_args)
    print("✅ Prosecutor rebuttal ready")
    

    # Defender final response
    defender_final = defender_respond(facts, precedent_analysis, prosecutor_rebuttal)
    print("✅ Defender final response ready")
    

    # STEP 4: Judge Agent
    print("\n👨‍⚖️ [4/5] Judge Agent — delivering verdict...")
    verdict = judge_verdict(
        case_facts=facts,
        precedents=precedent_analysis,
        prosecutor_arguments=prosecutor_args,
        defender_arguments=defender_args,
        prosecutor_rebuttal=prosecutor_rebuttal,
        defender_response=defender_final
    )
    print("✅ Verdict delivered")
    

    # STEP 5: Format output
    print("\n📄 [5/5] Formatting results...")
    

    result = {
        "facts": facts,
        "precedents": precedents,
        "precedent_analysis": precedent_analysis,
        "debate": {
            "prosecutor_arguments": prosecutor_args,
            "defender_arguments": defender_args,
            "prosecutor_rebuttal": prosecutor_rebuttal,
            "defender_final_response": defender_final
        },
        "verdict": verdict,
        "formatted_verdict": format_verdict(verdict)
    }
    

    print(result['formatted_verdict'])
    return result



   # Quick test with a sample case from the dataset

if __name__ == "__main__":
    from datasets import load_dataset
    
    ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
    test_case = ds['test'][0]
    
    result = run_pipeline(test_case['text'])