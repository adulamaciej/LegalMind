from rag.indexer import get_collection, index_cases
from rag.retriever import find_similar_cases
from config import call_claude, MODEL, labels_to_articles


def build_precedent_query(facts: dict) -> str:
    """Build a retrieval query from the extracted facts rather than raw paragraphs.

    The raw opening paragraphs of an ECHR case are mostly procedural; the facts
    summary, key events and alleged violations are the discriminative signal.
    """
    parts = [facts.get("summary", "")]
    parts += facts.get("key_events", []) or []
    parts += facts.get("alleged_violations", []) or []
    return " ".join(p for p in parts if p).strip()


class PrecedentAgent:
    
    """
    Legal Researcher Agent — finds similar ECHR cases using RAG.
    Uses ChromaDB as vector store via indexer and retriever modules.
    """

    def __init__(self, collection_name: str = "echr_cases"):
        self.collection = get_collection(collection_name=collection_name)

    def index_cases(self, cases: list[dict]) -> None:
        """Index cases into ChromaDB."""
        index_cases(cases, self.collection)

    def find_precedents(
        self, case_facts: list[str], n_results: int = 3, query_text: str | None = None
    ) -> list[dict]:
        """
        Find similar cases to given facts.
        Input: list of fact paragraphs, optionally an explicit query string
        Output: list of similar cases
        """
        return find_similar_cases(case_facts, self.collection, n_results, query_text=query_text)

    def analyze_precedents(self, case_facts: list[str], precedents: list[dict]) -> str:
        """
        Use LLM to analyze found precedents.
        Output: string with precedent analysis
        """
        facts_text = " ".join(case_facts[:5])
        precedents_text = "\n\n".join([
            f"Precedent {i+1} (violated articles: {labels_to_articles(p['labels'])}, "
            f"similarity: {p['similarity']:.3f}):\n{p['text'][:1500]}"
            for i, p in enumerate(precedents)
        ])

        prompt = f"""You are a legal researcher at the European Court of Human Rights.

CURRENT CASE FACTS:
{facts_text}

SIMILAR PRECEDENTS FOUND:
{precedents_text}

Analyze how these precedents are relevant to the current case:
- What articles were violated in similar cases?
- How should these precedents influence the current case analysis?

Ground rules:
- Base your analysis ONLY on the precedent excerpts and current case facts shown above.
- Do not use outside knowledge of ECHR case law or case names.
- Do not assert anything about a precedent that is not present in its excerpt.

Provide a concise analysis in 3-4 sentences."""

        try:
            return call_claude(prompt, model=MODEL, max_tokens=500)
        except Exception as e:
            print(f"Precedent analysis failed: {e}")
            return "Precedent analysis could not be completed due to a technical error."