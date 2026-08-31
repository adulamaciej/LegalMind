import chromadb


# How many paragraphs go into one embedded chunk, and how many paragraphs of a
# case we bother indexing at all. Small chunks keep each vector inside MiniLM's
# ~256 word-piece window so the substantive facts actually get embedded instead
# of being truncated away behind the procedural intro.
CHUNK_SIZE = 3
MAX_PARAGRAPHS = 40
MIN_CHUNK_WORDS = 15


# ChromaDB database on a disk
def get_collection(path: str = "./data/chroma", collection_name: str = "echr_cases"):

    """Initialize and return ChromaDB collection."""

    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def chunk_case(paragraphs: list[str]) -> list[str]:
    """Split a case's factual paragraphs into small consecutive chunks.

    Caps at the first MAX_PARAGRAPHS paragraphs and drops chunks too short to
    carry meaning (usually procedural fragments).
    """
    paragraphs = paragraphs[:MAX_PARAGRAPHS]
    chunks = []
    for start in range(0, len(paragraphs), CHUNK_SIZE):
        chunk_text = "\n".join(paragraphs[start:start + CHUNK_SIZE]).strip()
        if len(chunk_text.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk_text)
    return chunks


# Preparing and adding ECHR cases to ChromaDB
def index_cases(cases: list[dict], collection, batch_size: int = 500) -> None:
    """
    Index ECHR cases into ChromaDB, one document per paragraph chunk.
    Input: list of cases from dataset, ChromaDB collection
    """
    documents = []
    metadatas = []
    ids = []

    for i, case in enumerate(cases):
        labels = str(case['labels'])
        for chunk_idx, chunk_text in enumerate(chunk_case(case['text'])):
            documents.append(chunk_text)
            metadatas.append({"case_id": i, "labels": labels})
            ids.append(f"case_{i}_c{chunk_idx}")

    total = len(documents)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end]
        )
        print(f"Indexed {end}/{total} chunks...")

    print(f"✅ Finished indexing {total} chunks from {len(cases)} cases into ChromaDB.")


# Indexes training cases into ChromaDB
if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("coastalcph/lex_glue", "ecthr_a")
    collection = get_collection()
    index_cases(list(ds['train']), collection)
