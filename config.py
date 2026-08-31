import ast
import json
import os
from anthropic import Anthropic, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


ARTICLE_CODES = ['2', '3', '5', '6', '8', '9', '10', '11', '14', 'P1-1']  # dataset label index → article code


# Model configuration
MODEL = os.getenv("LEGALMIND_MODEL", "claude-haiku-4-5-20251001")

# Model used to grade the pipeline's own output during evaluation. Kept separate from
# MODEL so evaluation quality stays fixed regardless of which model the pipeline runs.
JUDGE_MODEL = os.getenv("LEGALMIND_JUDGE_MODEL", "claude-sonnet-5")


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

def extract_text(response):
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def labels_to_articles(labels_str: str) -> list[str]:
    """Convert a stored '[0, 5]' label-index string back into article codes."""
    try:
        indices = ast.literal_eval(labels_str)
        return [ARTICLE_CODES[i] for i in indices]
    except (ValueError, SyntaxError, IndexError, TypeError):
        return []


def parse_json_response(text_response: str) -> dict:
    """Strip any markdown fence the model wrapped the JSON in, then parse it."""
    text_response = text_response.strip()
    if text_response.startswith("```"):
        text_response = text_response.split("```")[1]
    if text_response.startswith("json"):
        text_response = text_response[4:]
    text_response = text_response.strip()

    try:
        return json.loads(text_response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}\nRaw response: {text_response}")
        raise


_client = Anthropic()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError, InternalServerError))
)
def call_claude(prompt: str, model: str, max_tokens: int) -> str:
    """Calls Claude with retry/backoff on transient failures. Returns extracted text."""
    response = _client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return extract_text(response)

