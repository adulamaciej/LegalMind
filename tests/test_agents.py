import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import extract_text, ARTICLE_CODES, ARTICLES_MAP
from evaluation.evaluation import evaluate_verdict


# --- extract_text tests ---

class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeThinkingBlock:
    def __init__(self):
        self.type = "thinking"


class FakeResponse:
    def __init__(self, content):
        self.content = content


def test_extract_text_returns_text_block():
    response = FakeResponse([FakeTextBlock("hello world")])
    assert extract_text(response) == "hello world"


def test_extract_text_skips_thinking_block():
    response = FakeResponse([FakeThinkingBlock(), FakeTextBlock("actual answer")])
    assert extract_text(response) == "actual answer"


def test_extract_text_returns_empty_if_no_text_block():
    response = FakeResponse([FakeThinkingBlock()])
    assert extract_text(response) == ""


# --- evaluate_verdict tests ---

def test_evaluate_verdict_exact_match():
    verdict = {"violated_articles": ["6", "8"]}
    result = evaluate_verdict(verdict, ["6", "8"])
    assert result["exact_match"] is True
    assert result["partial_match"] is True


def test_evaluate_verdict_partial_match_only():
    verdict = {"violated_articles": ["6", "9"]}
    result = evaluate_verdict(verdict, ["6", "8"])
    assert result["exact_match"] is False
    assert result["partial_match"] is True


def test_evaluate_verdict_no_match():
    verdict = {"violated_articles": ["9"]}
    result = evaluate_verdict(verdict, ["6"])
    assert result["exact_match"] is False
    assert result["partial_match"] is False


def test_evaluate_verdict_correctly_predicts_no_violation():
    verdict = {"violated_articles": []}
    result = evaluate_verdict(verdict, [])
    assert result["exact_match"] is True
    assert result["partial_match"] is True


# --- ARTICLE_CODES / ARTICLES_MAP consistency ---

def test_article_codes_map_consistency():
    for code in ARTICLE_CODES:
        assert code in ARTICLES_MAP, f"{code} missing from ARTICLES_MAP"


# --- judge_agent hallucination filtering (mocked, no API call) ---

@patch("agents.judge_agent.call_claude")
def test_judge_verdict_filters_hallucinated_codes(mock_call_claude):
    from agents.judge_agent import judge_verdict

    mock_call_claude.return_value = '{"violation": true, "violated_articles": ["6", "13"], "confidence_score": 80, "reasoning": "test reasoning with legal representation issues", "key_factors": []}'

    result = judge_verdict(
        case_facts={},
        precedents="",
        prosecutor_arguments="",
        defender_arguments="",
        prosecutor_rebuttal="",
        defender_response=""
    )

    assert "13" not in result["violated_articles"]
    assert "6" in result["violated_articles"]
    assert "13" in result["filtered_hallucinated_codes"]


@patch("agents.judge_agent.call_claude")
def test_judge_verdict_flags_low_confidence(mock_call_claude):
    from agents.judge_agent import judge_verdict

    mock_call_claude.return_value = '{"violation": true, "violated_articles": ["6"], "confidence_score": 50, "reasoning": "test reasoning with legal representation issues", "key_factors": []}'

    result = judge_verdict(
        case_facts={},
        precedents="",
        prosecutor_arguments="",
        defender_arguments="",
        prosecutor_rebuttal="",
        defender_response=""
    )

    assert result["low_confidence"] is True


@patch("agents.judge_agent.call_claude")
def test_judge_verdict_does_not_flag_high_confidence(mock_call_claude):
    from agents.judge_agent import judge_verdict

    mock_call_claude.return_value = '{"violation": true, "violated_articles": ["6"], "confidence_score": 90, "reasoning": "test reasoning with legal representation issues", "key_factors": []}'

    result = judge_verdict(
        case_facts={},
        precedents="",
        prosecutor_arguments="",
        defender_arguments="",
        prosecutor_rebuttal="",
        defender_response=""
    )

    assert result["low_confidence"] is False


# --- retrieval quality (pure, no API) ---
# ARTICLE_CODES index → code: 0='2', 1='3', 2='5', 3='6', 4='8'
# Labels are stored as stringified index lists, matching rag/indexer.py

def test_evaluate_retrieval_precision():
    from evaluation.rag_metrics import evaluate_retrieval

    precedents = [
        {"id": "case_1", "labels": "[3]", "similarity": 0.9},   # Article 6 — shared
        {"id": "case_2", "labels": "[0]", "similarity": 0.7},   # Article 2 — not shared
    ]
    result = evaluate_retrieval(["6", "8"], precedents)

    assert result["precision_at_k"] == 0.5
    assert result["hits"][0]["relevant"] is True
    assert result["hits"][1]["relevant"] is False


def test_evaluate_retrieval_no_overlap():
    from evaluation.rag_metrics import evaluate_retrieval

    precedents = [
        {"id": "case_1", "labels": "[0]", "similarity": 0.9},   # Article 2
        {"id": "case_2", "labels": "[1]", "similarity": 0.8},   # Article 3
    ]
    result = evaluate_retrieval(["6"], precedents)

    assert result["precision_at_k"] == 0.0
    assert all(h["overlap"] == 0.0 for h in result["hits"])


# --- precedent analysis grounding ---

def test_check_code_grounding_flags_ungrounded():
    from evaluation.rag_metrics import check_code_grounding

    precedents = [{"id": "case_1", "labels": "[3]", "similarity": 0.9}]  # Article 6 only
    analysis = "The precedents establish violations of Article 6 and Article 8."

    assert check_code_grounding(analysis, precedents) == ["8"]


def test_check_code_grounding_ignores_substring_match():
    from evaluation.rag_metrics import check_code_grounding

    precedents = [{"id": "case_1", "labels": "[0]", "similarity": 0.9}]  # Article 2 only
    analysis = "Decided in 1968, the case concerned Article 16 of an unrelated treaty."

    # Naive substring matching would find '6' inside 'Article 16' and '1968',
    # and '2' inside '1968', reporting phantom ungrounded codes.
    assert check_code_grounding(analysis, precedents) == []


# --- retriever chunk -> case aggregation (no ChromaDB) ---

class FakeCollection:
    def __init__(self, result):
        self._result = result

    def query(self, query_texts, n_results):
        return self._result


def test_find_similar_cases_collapses_chunks_to_cases():
    from rag.retriever import find_similar_cases

    # Two chunks from case 7, one from case 3 — best-first order.
    result = {
        "documents": [["c7 chunk a", "c3 chunk", "c7 chunk b"]],
        "distances": [[0.1, 0.3, 0.4]],
        "metadatas": [[
            {"case_id": 7, "labels": "[3]"},
            {"case_id": 3, "labels": "[0]"},
            {"case_id": 7, "labels": "[3]"},
        ]],
    }
    cases = find_similar_cases([], FakeCollection(result), n_results=3)

    assert [c["id"] for c in cases] == ["case_7", "case_3"]
    assert cases[0]["similarity"] == 1 - 0.1  # best chunk of case 7
    assert "c7 chunk a" in cases[0]["text"] and "c7 chunk b" in cases[0]["text"]
    assert "_chunks" not in cases[0]


def test_find_similar_cases_respects_n_results():
    from rag.retriever import find_similar_cases

    result = {
        "documents": [["a", "b", "c"]],
        "distances": [[0.1, 0.2, 0.3]],
        "metadatas": [[
            {"case_id": 1, "labels": "[0]"},
            {"case_id": 2, "labels": "[1]"},
            {"case_id": 3, "labels": "[2]"},
        ]],
    }
    cases = find_similar_cases([], FakeCollection(result), n_results=2)
    assert [c["id"] for c in cases] == ["case_1", "case_2"]


@patch("evaluation.rag_metrics.call_claude")
def test_score_faithfulness_parses_ratio(mock_call_claude):
    from evaluation.rag_metrics import score_faithfulness

    mock_call_claude.return_value = '{"claims": [{"claim": "a", "verdict": "supported", "reason": ""}, {"claim": "b", "verdict": "supported", "reason": ""}, {"claim": "c", "verdict": "unsupported", "reason": ""}]}'

    result = score_faithfulness("analysis text", [{"labels": "[3]", "text": "precedent text"}])

    assert result["n_claims"] == 3
    assert result["n_supported"] == 2
    assert result["faithfulness"] == 2 / 3


@patch("evaluation.rag_metrics.call_claude")
def test_score_faithfulness_salvages_truncated_json(mock_call_claude):
    from evaluation.rag_metrics import score_faithfulness

    # Response cut off mid-object, as happens when the judge hits the token ceiling.
    mock_call_claude.return_value = (
        '{"claims": [{"claim": "a", "verdict": "supported"}, '
        '{"claim": "b", "verdict": "unsupported"}, {"claim": "c", "verdict": "'
    )

    result = score_faithfulness("analysis text", [{"labels": "[3]", "text": "precedent text"}])

    assert result["faithfulness"] == 1 / 2
    assert result["n_claims"] == 2
    assert result["truncated"] is True


def test_evaluate_retrieval_undefined_for_no_violation_case():
    from evaluation.rag_metrics import evaluate_retrieval

    precedents = [
        {"id": "case_1", "labels": "[3]", "similarity": 0.9},
        {"id": "case_2", "labels": "[0]", "similarity": 0.7},
    ]
    result = evaluate_retrieval([], precedents)

    assert result["precision_at_k"] is None
    assert len(result["hits"]) == 2
    assert all(h["relevant"] is False for h in result["hits"])