def find_similar_cases(query_paragraphs: list[str], collection, n_results: int = 3) -> list[dict]:

    """
    Find similar ECHR cases using ChromaDB vector search.
    Input: list of case paragraphs, ChromaDB collection
    Output: list of similar cases with similarity scores
    """
    
    query = " ".join(query_paragraphs[:10])

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    precedents = []
    for i in range(len(results['documents'][0])):
        precedents.append({
            "id": results['ids'][0][i],
            "text": results['documents'][0][i],
            "labels": results['metadatas'][0][i]['labels'],
            "similarity": 1 - results['distances'][0][i]
        })

    return precedents