from __future__ import annotations

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from rag import (
    CACHE_CHUNKS,
    CACHE_EMBEDDINGS,
    EMBEDDING_MODEL,
    document_fingerprints,
    load_chunks,
)


def main() -> None:
    chunks = load_chunks()
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = np.asarray(
        model.encode(
            [chunk.text for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )

    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "documents": document_fingerprints(),
        "chunks": [chunk.__dict__ for chunk in chunks],
    }
    np.savez_compressed(CACHE_EMBEDDINGS, embeddings=embeddings)
    CACHE_CHUNKS.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(chunks)} embeddings to {CACHE_EMBEDDINGS.name}.")


if __name__ == "__main__":
    main()
