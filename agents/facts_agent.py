import json
from anthropic import Anthropic
from config import ARTICLES_MAP, ARTICLE_CODES

client = Anthropic()


def summarize_long_case(paragraphs: list[str], chunk_size: int = 40) -> list[str]:
    """
    For cases too long to fit directly, summarize in chunks
    to preserve information from the entire case, not just the start.
    """
    if len(paragraphs) <= 50:
        return paragraphs  # no summarization needed

    # split into chunks
    chunks = [paragraphs[i:i+chunk_size] for i in range(0, len(paragraphs), chunk_size)]
    
    summarized_chunks = []
    for chunk in chunks:
        text = "\n".join(chunk)
        prompt = f"""Summarize the following excerpt from an ECHR case's factual background.
Preserve all specific events, dates, and details relevant to potential human rights violations.
Do not omit procedural steps, medical/legal findings, or testimony.

EXCERPT:
{text}

Provide a concise but complete summary (aim for ~30% of original length):"""

        response = client.messages.create(
            model="claude-sonnet-5",  # fix this model name too
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        summarized_chunks.append(response.content[0].text)

    return summarized_chunks  # now feed this into extract_facts()


def extract_facts(case_paragraphs: list[str]) -> dict:
    case_paragraphs = summarize_long_case(case_paragraphs)
    
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
        model="claude-sonnet-5",
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
    return result


if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
    case = ds['train'][0]
    facts = extract_facts(case['text'])
    print("=== EXTRACTED FACTS ===")
    print(json.dumps(facts, indent=2))
    print("\n=== GROUND TRUTH ===")
    print([ARTICLES_MAP[ARTICLE_CODES[l]] for l in case['labels']])

