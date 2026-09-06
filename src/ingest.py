from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
import ollama
from chromadb.api.types import Metadata, PyEmbedding

from .processing.config import ChunkConfig, EmbeddingConfig

class CONFIG:
    chunk_config = ChunkConfig()
    embedding_config = EmbeddingConfig()

class PATH:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    CORPUS_ROOT = PROJECT_ROOT / "corpus" / "public"
    DATABASE_PATH = PROJECT_ROOT / "data" / "chroma"

class OLLAMA:
    COLLECTION_NAME = "palimpsest_passages"
    EMBEDDING_MODEL = "embeddinggemma"

def normalize_text(text: str) -> str:
    paragraphs = [
        " ".join(line.split()) for line in text.splitlines() if line.strip()
    ]

    return "\n\n".join(paragraphs)

def chunk_text(
        text: str,
        config: ChunkConfig
) -> list[str]:

    chunk_size = config.size
    overlap = config.overlap

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size")


    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks

def create_chunk_id(
        source_id: str,
        chunk_number: int,
        text: str
) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    return f"{source_id}:{chunk_number}:{digest}"


def load_source(source_directory: Path) -> tuple[str, dict[str, Any]]:
    text_path = source_directory / "source.txt"
    metadata_path = source_directory / "metadata.json"

    if not text_path.exists():
        raise FileNotFoundError(f"Missing source: {text_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    text = text_path.read_text(encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))


    required_fields = {"id", "title", "author", "tradition", "period", "genre", "source_type"}

    missing_fields = required_fields - metadata.keys()

    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Missing required metadata fields: {missing}")

    return normalize_text(text), metadata

def get_collection() -> chromadb.Collection:
    PATH.DATABASE_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(PATH.DATABASE_PATH)
    )

    return client.get_or_create_collection(
        name=OLLAMA.COLLECTION_NAME,
        metadata={"description": ("Passages indexed by the Palimpsest research system")}
    )


def embed_batch(texts: list[str]) -> list[PyEmbedding]:
    response = ollama.embed(
        model=OLLAMA.EMBEDDING_MODEL,
        input=texts,
    )

    embeddings: list[PyEmbedding] = [list(vector) for vector in response.embeddings]
    return embeddings

def ingest_source(collection: chromadb.Collection,source_directory: Path, chunk_config: ChunkConfig, embedding_config: EmbeddingConfig) -> int:
    text, source_metadata = load_source(source_directory)
    chunks = chunk_text(text, chunk_config)

    source_id = str(source_metadata["id"])

    relative_path = source_directory.relative_to(PATH.CORPUS_ROOT)
    
    print(
        f"Ingesting {source_metadata['title']} "
        f"[{relative_path}] as {len(chunks)} chunks..."
    )

    inserted = 0


    for batch_start in range(0, len(chunks), embedding_config.batch_size):
        batch = chunks[batch_start:batch_start + embedding_config.batch_size]

        embeddings = embed_batch(batch)

        ids: list[str] = []
        metadatas: list[Metadata] = []

        for offset, chunk in enumerate(batch):
            chunk_number = batch_start + offset

            ids.append(create_chunk_id(
                source_id=source_id,
                chunk_number=chunk_number,
                text=chunk,
            ))

            metadatas.append(
                {
                    **source_metadata,
                    "chunk_number": chunk_number,
                    "source_path": str(source_directory.relative_to(PATH.CORPUS_ROOT)),
                }
            )

        collection.upsert(
            ids=ids,
            documents=batch,
            embeddings=embeddings,
            metadatas=metadatas
        )

        inserted += len(batch)

    return inserted

def discover_sources(corpus_root: Path) -> list[Path]:
    source_directories: list[Path] = []

    for metadata_path in corpus_root.rglob("metadata.json"):
        source_directory = metadata_path.parent
        source_path = source_directory / "source.txt"
        
        if not source_path.exists():
            raise FileNotFoundError(f"Found metadata without source.txt:\n\t{source_path}")

        source_directories.append(source_directory)

    return sorted(source_directories)

def main() -> None: 
    collection = get_collection()
    chunk_config = ChunkConfig()
    embedding_config = EmbeddingConfig()

    source_directories = discover_sources(PATH.CORPUS_ROOT)

    if not source_directories:
        raise RuntimeError(f"No sources found in {PATH.CORPUS_ROOT}")

    total = 0

    for source_directory in source_directories:
        total += ingest_source(collection=collection, source_directory=source_directory, chunk_config=chunk_config, embedding_config=embedding_config)

    print(f"Ingestion complete: {total} passages indexed.")

if __name__ == "__main__":
    main()