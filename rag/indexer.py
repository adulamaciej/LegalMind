import chromadb


def get_collection(path: str = "./data/chroma", collection_name: str = "echr_cases"):

    """Initialize and return ChromaDB collection."""

    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection


def index_cases(cases: list[dict], collection) -> None:

    """
    Index ECHR cases into ChromaDB.
    Input: list of cases from dataset, ChromaDB collection
    """
    
    documents = []
    metadatas = []
    ids = []

    for i, case in enumerate(cases):
        text = " ".join(case['text'][:10])
        documents.append(text)
        metadatas.append({"labels": str(case['labels'])})
        ids.append(f"case_{i}")

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Indexed {len(cases)} cases into ChromaDB.")


if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
    collection = get_collection()
    index_cases(list(ds['train']), collection)