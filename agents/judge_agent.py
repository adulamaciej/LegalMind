import json
from anthropic import Anthropic
from config import ARTICLES_MAP
from config import extract_text
from config import MODEL, ARTICLE_CODES

client = Anthropic()


def judge_verdict(
    case_facts: dict,
    precedents: str,
    prosecutor_arguments: str,
    defender_arguments: str,
    prosecutor_rebuttal: str,
    defender_response: str
) -> dict:
    transcript = f"""
=== CASE FACTS ===
{json.dumps(case_facts, indent=2)}

=== PRECEDENTS ===
{precedents}

=== PROSECUTOR ARGUMENTS ===
{prosecutor_arguments}

=== DEFENDER ARGUMENTS ===
{defender_arguments}

=== PROSECUTOR REBUTTAL ===
{prosecutor_rebuttal}

=== DEFENDER FINAL RESPONSE ===
{defender_response}
"""
    
    # The model hallucinated and used article 14 which does not exist in this dataset so explicitly listing allowed articles here
    allowed_articles = ", ".join(f"'{code}'" for code in ARTICLE_CODES)
    prompt = f"""You are a judge at the European Court of Human Rights.
You have read the full transcript of the debate between prosecutor and defender.

FULL TRANSCRIPT:
{transcript}

Based on the facts, precedents, and arguments presented, deliver your verdict.

IMPORTANT: "violated_articles" must ONLY contain codes from this exact list: {allowed_articles}
Do not invent or use any article code outside this list.

Return ONLY a JSON object with these fields:
{{
    "violation": true or false,
    "violated_articles": ["list of violated ECHR articles, using ONLY codes from: {allowed_articles}"],
    "confidence_score": 0-100,
    "reasoning": "detailed explanation of the verdict",
    "key_factors": ["list of key factors that influenced the decision"]
}}

Return ONLY the JSON, no other text."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    text_response = extract_text(response).strip()
    if text_response.startswith("```"):

        text_response = text_response.split("```")[1]
    if text_response.startswith("json"):
        text_response = text_response[4:]
    text_response = text_response.strip()

    try:
        result = json.loads(text_response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}\nRaw response: {text_response}")
        raise
    return result
