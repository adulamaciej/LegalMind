# Searches ChromaDB for similar cases, returns them ranked

# How many chunk hits to pull before collapsing them down to distinct cases.
CHUNK_FETCH = 30
# How many of a case's matching chunks to keep as context for downstream analysis.
CHUNKS_PER_CASE = 2


def find_similar_cases(
    query_paragraphs: list[str],
    collection,
    n_results: int = 3,
    query_text: str | None = None,
) -> list[dict]:

    """
    Find similar ECHR cases using ChromaDB vector search over paragraph chunks.

    The index holds one document per 3-paragraph chunk, so a query first matches
    chunks; we then collapse those to distinct cases, scoring each case by its
    best-matching chunk.

    Input: case paragraphs (fallback query), ChromaDB collection, optionally an
    explicit query string (e.g. the extracted-facts summary).
    Output: list of similar cases with similarity scores and their top chunk text.
    """

    query = query_text or " ".join(query_paragraphs[:10])

    results = collection.query(
        query_texts=[query],
        n_results=CHUNK_FETCH,
    )

    # Collapse chunk hits to cases, keeping each case's best similarity and its
    # top few matching chunks (results come back sorted best-first).
    cases: dict = {}
    for i in range(len(results['documents'][0])):
        meta = results['metadatas'][0][i]
        case_id = meta['case_id']
        similarity = 1 - results['distances'][0][i]

        case = cases.get(case_id)
        if case is None:
            cases[case_id] = {
                "id": f"case_{case_id}",
                "text": results['documents'][0][i],
                "labels": meta['labels'],
                "similarity": similarity,
                "_chunks": 1,
            }
        elif case["_chunks"] < CHUNKS_PER_CASE:
            case["text"] += "\n\n" + results['documents'][0][i]
            case["similarity"] = max(case["similarity"], similarity)
            case["_chunks"] += 1

    ranked = sorted(cases.values(), key=lambda c: c["similarity"], reverse=True)
    for c in ranked:
        del c["_chunks"]
    return ranked[:n_results]
