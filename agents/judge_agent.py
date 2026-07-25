import json
from anthropic import Anthropic
from config import ARTICLES_MAP

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

    prompt = f"""You are a judge at the European Court of Human Rights.
You have read the full transcript of the debate between prosecutor and defender.

FULL TRANSCRIPT:
{transcript}

Based on the facts, precedents, and arguments presented, deliver your verdict.

Return ONLY a JSON object with these fields:
{{
    "violation": true or false,
    "violated_articles": ["list of violated ECHR articles, e.g. '6', '8', 'P1-1'"],
    "confidence_score": 0-100,
    "reasoning": "detailed explanation of the verdict",
    "key_factors": ["list of key factors that influenced the decision"]
}}

Return ONLY the JSON, no other text."""

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    text_response = response.content[0].text.strip()
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


def format_verdict(verdict: dict) -> str:
    violation_str = "VIOLATION FOUND" if verdict['violation'] else "NO VIOLATION"
    articles = ", ".join([
        ARTICLES_MAP.get(a, a) for a in verdict.get('violated_articles', [])
    ])
    
    output = f"""
╔════════════════════════════════════════╗
║           ECHR COURT VERDICT           ║
╚════════════════════════════════════════╝

VERDICT: {violation_str}
VIOLATED ARTICLES: {articles if articles else 'None'}
CONFIDENCE SCORE: {verdict['confidence_score']}%

REASONING:
{verdict['reasoning']}

KEY FACTORS:
{chr(10).join([f"• {f}" for f in verdict.get('key_factors', [])])}
"""
    return output