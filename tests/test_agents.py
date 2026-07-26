import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import extract_text, ARTICLE_CODES, ARTICLES_MAP
from evaluation import evaluate_verdict


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