import os


ARTICLE_CODES = ['2', '3', '5', '6', '8', '9', '10', '11', '14', 'P1-1']  # dataset label index → article code


# Model configuration
MODEL = os.getenv("LEGALMIND_MODEL", "claude-haiku-4-5-20251001")


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