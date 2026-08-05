from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import ollama

class PATH:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATABASE_PATH = PROJECT_ROOT / "data" / "chroma"

class OLLAMA:
    COLLECTION_NAME = "palimpsest_passages"
    EMBEDDING_MODEL = "embeddinggemma"

def search_passages(
        query: str,
        result_count: int = 5
) -> list[dict[str. Any]]:
    if not query.strip():
        raise ValueError("Query must not be empty")

    client = chromadb.PersistentClient(path=PATH.DATABASE_PATH)
    collection = client.get_collection(name=OLLAMA.COLLECTION_NAME)
    embedding_response = ollama.embed(model=OLLAMA.EMBEDDING_MODEL, input=query)

    query_embedding = embedding_response.embeddings[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=result_count,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    passages: list[dict[str, Any]] = []

    for rank, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        passages.append(
            {
                "rank": rank,
                "document": document,
                "metadata": metadata,
                "distance": distance
            }
        )

    return passages


def main() -> None:
    query = input("Search Palimpsest: ").strip()
    passages = search_passages(query)

    for passage in passages:
        metadata = passage["metadata"]

        print()
        print("-" * 80)
        print(
            f"[{passage['rank']}]"
            f"{metadata['title']} - {metadata['author']}"
        )
        print(
            f"Chunk: {metadata['chunk_number']} | "
            f"Distance: {passage['distance']:.4f}"
        )
        print("-" * 80)
        print(passage["document"])

if __name__ == "__main__":
    main()