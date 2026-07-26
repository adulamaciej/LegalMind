import json
from anthropic import Anthropic
from config import ARTICLES_MAP
from config import extract_text
from config import MODEL, ARTICLE_CODES
from collections import defaultdict

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

Example of CORRECT format: ["6", "8"]
Example of INCORRECT format: ["13", "15"] — these codes do not exist and must never be used


Only include an article in "violated_articles" if there is strong, direct evidence
of a violation. Do not include articles based on speculation, tangential connections,
or weak circumstantial evidence. When in doubt, exclude the article rather than include it.

Most ECHR cases involve a single article violation. Multiple violations should only
be reported when there is clear, independent evidence for each one.

Before finalizing your list of violated articles, review EACH article you plan to
include one final time and ask: "Is this article central to the core violation, or
is it a secondary/tangential issue mentioned only in passing?" Remove any article
that is not central to the primary violation being alleged.


Note: Article 6 (fair trial) is common in ECHR case law, but should not be assumed by
default. Base your decision on the actual strength of the evidence presented in the
facts and arguments — if the case clearly involves issues like denial of legal
representation, non-disclosure of evidence, or lack of an impartial tribunal, include
Article 6. Do not omit it simply because it is a frequently-cited article; the goal is
accuracy in both directions, not caution for its own sake.

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
    result['violated_articles'] = [a for a in result.get('violated_articles', []) if a in ARTICLE_CODES]


    CONFIDENCE_THRESHOLD = 70
    if result.get('confidence_score', 0) < CONFIDENCE_THRESHOLD:
        print(f"Warning: low confidence ({result.get('confidence_score')}%) — verdict may be unreliable")

    FAIR_TRIAL_KEYWORDS = ["legal representation", "disclosure", "impartial", "adversarial", "equality of arms"]
    if '6' in result.get('violated_articles', []):
        reasoning_lower = result.get('reasoning', '').lower()
        if not any(kw in reasoning_lower for kw in FAIR_TRIAL_KEYWORDS):
            print("Warning: Article 6 flagged without specific fair-trial keyword evidence")

    return result
