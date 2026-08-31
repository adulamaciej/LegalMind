# Metrics for the RAG stage: how good is retrieval, and does the precedent
# analysis stay grounded in what was actually retrieved?

import re

from config import ARTICLE_CODES, JUDGE_MODEL, call_claude, labels_to_articles, parse_json_response


# "Article 6", "Articles 6, 8 and 14", "Article P1-1" — a bare code is matched only for
# P1-1, which is unambiguous. Matching bare digits would make "6" hit inside "16" or "1968".
_ARTICLE_MENTION = re.compile(
    r"\bArticles?\s+((?:P1-1|\d{1,2})(?:\s*(?:,|and|&)\s*(?:P1-1|\d{1,2}))*)",
    re.IGNORECASE,
)
_CODE_IN_LIST = re.compile(r"P1-1|\d{1,2}", re.IGNORECASE)
_BARE_P1_1 = re.compile(r"\bP1-1\b", re.IGNORECASE)


def evaluate_retrieval(query_labels: list[str], precedents: list[dict]) -> dict:
    """
    Score retrieval quality against the dataset's own article labels.

    A precedent counts as relevant if it shares at least one violated article with the
    query case — no LLM judgement needed, since every case carries ground-truth labels.

    Keeping `similarity` next to `relevant` is the point: correlating them shows whether
    pure semantic similarity actually predicts legal relevance.

    For a no-violation query case there is no article any precedent could share, so
    `precision_at_k` is returned as None (undefined, not zero) and such cases are
    excluded from the aggregate.
    """
    query_set = set(query_labels)

    hits = []
    for p in precedents:
        precedent_set = set(labels_to_articles(p.get('labels', '')))
        shared = query_set & precedent_set
        union = query_set | precedent_set
        hits.append({
            "id": p.get('id'),
            "relevant": bool(shared),
            "overlap": len(shared) / len(union) if union else 0.0,
            "similarity": p.get('similarity'),
        })

    if not query_set:
        precision_at_k = None
    elif hits:
        precision_at_k = sum(h["relevant"] for h in hits) / len(hits)
    else:
        precision_at_k = 0.0

    return {
        "precision_at_k": precision_at_k,
        "hits": hits,
    }


def mentioned_articles(text: str) -> set[str]:
    """Article codes the text actually cites, restricted to this dataset's vocabulary."""
    found = set()
    for match in _ARTICLE_MENTION.finditer(text):
        found.update(code.upper() for code in _CODE_IN_LIST.findall(match.group(1)))
    if _BARE_P1_1.search(text):
        found.add('P1-1')
    return {code for code in found if code in ARTICLE_CODES}


def check_code_grounding(analysis_text: str, precedents: list[dict]) -> list[str]:
    """
    Return article codes the analysis cites that no retrieved precedent supports.

    The free half of faithfulness: a code claimed in the analysis but absent from every
    retrieved precedent's labels is ungrounded, no judge call required.
    """
    grounded = {code for p in precedents for code in labels_to_articles(p.get('labels', ''))}
    return sorted(mentioned_articles(analysis_text) - grounded)


def score_faithfulness(analysis_text: str, precedents: list[dict], model: str = JUDGE_MODEL) -> dict:
    """
    LLM-as-judge faithfulness: what fraction of the analysis's claims are supported
    by the retrieved precedents?

    Covers the prose claims a regex cannot check ("this pattern strongly suggests the
    detention was arbitrary"). Article codes are already handled by check_code_grounding.
    """
    precedents_text = "\n\n".join([
        f"PRECEDENT {i+1} (violated articles: {labels_to_articles(p['labels'])}):\n{p['text'][:1500]}"
        for i, p in enumerate(precedents)
    ])

    prompt = f"""You are evaluating whether a legal analysis stays grounded in its source material.

RETRIEVED PRECEDENTS:
{precedents_text}

ANALYSIS TO EVALUATE:
{analysis_text}

Break the analysis into atomic factual claims — each a single, self-contained assertion.
For each claim, decide whether the RETRIEVED PRECEDENTS above support it.

A claim is "supported" only if the precedents provide direct evidence for it.
A claim is "unsupported" if it relies on outside knowledge, invents details not present
in the precedents, or overstates what the precedents show.
Judge only against the precedents shown — not against your own knowledge of ECHR case law.

Keep each "claim" string short (one clause). Return ONLY a JSON object with this shape:
{{
    "claims": [
        {{"claim": "the atomic claim", "verdict": "supported" or "unsupported"}}
    ]
}}

Return ONLY the JSON, no other text."""

    raw = ""
    try:
        raw = call_claude(prompt, model=model, max_tokens=8000)
        claims = parse_json_response(raw).get('claims', [])
        verdicts = [c.get('verdict') for c in claims]
        return _faithfulness_result(verdicts, claims=claims)
    except Exception as e:
        # A truncated response still carries usable verdicts — salvage them rather
        # than dropping the whole case from the sample.
        salvaged = re.findall(r'"verdict"\s*:\s*"(supported|unsupported)"', raw)
        if salvaged:
            print(f"Faithfulness JSON incomplete ({e}); salvaged {len(salvaged)} verdicts.")
            return _faithfulness_result(salvaged, truncated=True)
        print(f"Faithfulness judging failed: {e}")
        return {"faithfulness": None, "n_claims": 0, "n_supported": 0, "claims": [], "error": str(e)}


def _faithfulness_result(verdicts: list, claims: list | None = None, truncated: bool = False) -> dict:
    supported = sum(1 for v in verdicts if v == 'supported')
    result = {
        "faithfulness": supported / len(verdicts) if verdicts else None,
        "n_claims": len(verdicts),
        "n_supported": supported,
        "claims": claims if claims is not None else [{"verdict": v} for v in verdicts],
    }
    if truncated:
        result["truncated"] = True
    return result
