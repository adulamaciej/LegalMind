import chromadb


# ChromaDB database on a disk
def get_collection(path: str = "./data/chroma", collection_name: str = "echr_cases"):

    """Initialize and return ChromaDB collection."""

    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection


# Preparing and adding ECHR cases to ChromaDB
def index_cases(cases: list[dict], collection, batch_size: int = 500) -> None:
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

    total = len(documents)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end]
        )
        print(f"Indexed {end}/{total} cases...")

    print(f"✅ Finished indexing {total} cases into ChromaDB.")


# Indexes training cases into ChromaDB
if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
    collection = get_collection()
    index_cases(list(ds['train']), collection)