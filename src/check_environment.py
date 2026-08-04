from __future__ import annotations

import sys

import chromadb
import ollama


# GLOBAL VARS  
GENERATION_MODEL = "gemma3:4b"
EMBEDDING_MODEL = "embeddinggemma"

def main() -> None: 
    print(f"Python: {sys.version.split()[0]}")
    print(f"Chroma: {chromadb.__version__}")

    models_response = ollama.list()
    installed_models = { model.model for model in models_response.models}

    print("Installed Ollama models:" )
    for model_name in sorted(installed_models):
        print(f"  - {model_name}")

    generation_response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "user", 
                "content": (
                    "Reply with exactly: "
                    "Palimpsest generation check passed."
                ),
            }
        ],
    )

    print(generation_response.message.content)

    embedding_response = ollama.embed(
        model=EMBEDDING_MODEL,
        input="The unexamined life is not worth living."
    )

    vector = embedding_response.embeddings[0]

    print(f"Embedding dimensions: {len(vector)}")
    print("Palimpsest environment is operational.")

if __name__ == "__main__":
    main()