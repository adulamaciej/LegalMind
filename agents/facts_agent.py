import os
import json
from anthropic import Anthropic

client = Anthropic()

ARTICLES_MAP = {
    0: "Article 2 (right to life)",
    1: "Article 3 (prohibition of torture)",
    2: "Article 5 (right to liberty)",
    3: "Article 6 (right to fair trial)",
    4: "Article 8 (right to private/family life)",
    5: "Article 9 (freedom of thought)",
    6: "Article 10 (freedom of expression)",
    7: "Article 11 (freedom of assembly)",
    8: "Article 14 (prohibition of discrimination)",
    9: "P1-1 (protection of property)"
}

def extract_facts(case_paragraphs: list[str]) -> dict:
    if len(case_paragraphs) > 50:
        case_paragraphs = case_paragraphs[:50]
    
    text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(case_paragraphs)])
    
    prompt = f"""You are a legal clerk at the European Court of Human Rights.
Analyze the following case facts and extract structured information.

CASE FACTS:
{text}

Extract and return ONLY a JSON object with these fields:
{{
    "parties": {{
        "applicant": "description of applicant(s)",
        "respondent_state": "country being sued"
    }},
    "key_events": ["list of key events in chronological order"],
    "alleged_violations": ["list of potential human rights violations"],
    "relevant_articles": ["list of ECHR articles potentially violated"],
    "summary": "2-3 sentence summary of the case"
}}

Return ONLY the JSON, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    text_response = response.content[0].text.strip()
    if text_response.startswith("```"):
        text_response = text_response.split("```")[1]
        if text_response.startswith("json"):
            text_response = text_response[4:]
    text_response = text_response.strip()
    
    result = json.loads(text_response)
    text_response = text_response.strip()
    print("RAW RESPONSE:", text_response) 
    result = json.loads(text_response)
    return result


if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
    case = ds['train'][0]
    facts = extract_facts(case['text'])
    print("=== EXTRACTED FACTS ===")
    print(json.dumps(facts, indent=2))
    print("\n=== GROUND TRUTH ===")
    print([ARTICLES_MAP[l] for l in case['labels']])