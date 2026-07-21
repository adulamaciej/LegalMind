import os
import json
from anthropic import Anthropic

client = Anthropic()

ARTICLES_MAP = {
    "2": "Article 2 (right to life)",
    "3": "Article 3 (prohibition of torture)",
    "5": "Article 5 (right to liberty)",
    "6": "Article 6 (right to fair trial)",
    "8": "Article 8 (right to private/family life)",
    "9": "Article 9 (freedom of thought)",
    "10": "Article 10 (freedom of expression)",
    "11": "Article 11 (freedom of assembly)",
    "14": "Article 14 (prohibition of discrimination)",
    "P1-1": "P1-1 (protection of property)"
}

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
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = json.loads(response.content[0].text)
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