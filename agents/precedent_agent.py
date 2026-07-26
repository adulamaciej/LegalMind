from anthropic import Anthropic
from rag.indexer import get_collection, index_cases
from rag.retriever import find_similar_cases
from config import extract_text
from config import MODEL

client = Anthropic()


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

    def find_precedents(self, case_facts: list[str], n_results: int = 3) -> list[dict]:
        """
        Find similar cases to given facts.
        Input: list of fact paragraphs
        Output: list of similar cases
        """
        return find_similar_cases(case_facts, self.collection, n_results)

    def analyze_precedents(self, case_facts: list[str], precedents: list[dict]) -> str:
        """
        Use LLM to analyze found precedents.
        Output: string with precedent analysis
        """
        facts_text = " ".join(case_facts[:5])
        precedents_text = "\n".join([
            f"Precedent {i+1}: {p['text'][:200]}..."
            for i, p in enumerate(precedents)
        ])

        prompt = f"""You are a legal researcher at the European Court of Human Rights.

CURRENT CASE FACTS:
{facts_text}

SIMILAR PRECEDENTS FOUND:
{precedents_text}

Analyze how these precedents are relevant to the current case.
What articles were violated in similar cases?
How should these precedents influence the current case analysis?

Provide a concise analysis in 3-4 sentences."""

        try:
            response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
            return extract_text(response)
        except Exception as e:
            print(f"Precedent analysis failed: {e}")
            return "Precedent analysis could not be completed due to a technical error."


