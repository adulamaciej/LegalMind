from anthropic import Anthropic
from rag.indexer import get_collection, index_cases
from rag.retriever import find_similar_cases

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
            model="claude-sonnet-5",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
            return response.content[0].text
        except Exception as e:
            print(f"Precedent analysis failed: {e}")
            return "Precedent analysis could not be completed due to a technical error."



# Manual test: indexes 100 sample cases into ChromaDB, then sanity-checks precedent retrieval + analysis
if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("coastalcph/lex_glue", "ecthr_a")

    agent = PrecedentAgent()

    # Index first 100 cases as precedents
    cases = [ds['train'][i] for i in range(100)]
    agent.index_cases(cases)

    # Find precedents for a new case
    test_case = ds['test'][0]
    precedents = agent.find_precedents(test_case['text'])
    analysis = agent.analyze_precedents(test_case['text'], precedents)

    print("=== PRECEDENTS FOUND ===")
    for p in precedents:
        print(f"ID: {p['id']}, Similarity: {p['similarity']:.3f}")

    print("\n=== ANALYSIS ===")
    print(analysis)