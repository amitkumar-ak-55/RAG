from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"
CACHE_EMBEDDINGS = PROJECT_DIR / "embeddings.npz"
CACHE_CHUNKS = PROJECT_DIR / "chunks.json"
load_dotenv(PROJECT_DIR / ".env")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CHAT_MODEL = os.getenv("CHAT_MODEL", "minimax/minimax-m3.1-flash:free")
CHAT_MODEL_FALLBACKS = os.getenv(
    "CHAT_MODEL_FALLBACKS",
    "nvidia/nemotron-3.5-super-vl:free,dots-studio/dots-3-note-preview:free",
)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
TOP_K = 3
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.30"))


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_number: int


def load_chunks(documents_dir: Path = DOCUMENTS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(documents_dir.glob("*.md")):
        paragraphs = [part.strip() for part in path.read_text(encoding="utf-8").split("\n\n")]
        for chunk_number, paragraph in enumerate(paragraphs, start=1):
            if paragraph:
                chunks.append(Chunk(paragraph, path.name, chunk_number))
    if not chunks:
        raise FileNotFoundError(f"No Markdown documents found in {documents_dir}")
    return chunks


def document_fingerprints(documents_dir: Path = DOCUMENTS_DIR) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(documents_dir.glob("*.md"))
    }


def load_embedding_cache() -> tuple[list[Chunk], np.ndarray]:
    if not CACHE_EMBEDDINGS.exists() or not CACHE_CHUNKS.exists():
        raise FileNotFoundError(
            "Embedding cache is missing. Run '.venv\\Scripts\\python.exe RAG\\embed_documents.py' first."
        )

    metadata = json.loads(CACHE_CHUNKS.read_text(encoding="utf-8"))
    if metadata.get("embedding_model") != EMBEDDING_MODEL:
        raise RuntimeError(
            "Embedding model changed. Rebuild the cache with 'python RAG\\embed_documents.py'."
        )
    if metadata.get("documents") != document_fingerprints():
        raise RuntimeError(
            "Source documents changed. Rebuild the cache with 'python RAG\\embed_documents.py'."
        )

    chunks = [Chunk(**item) for item in metadata["chunks"]]
    embeddings = np.load(CACHE_EMBEDDINGS)["embeddings"]
    if len(chunks) != len(embeddings):
        raise RuntimeError("Embedding cache is inconsistent. Rebuild it with 'python RAG\\embed_documents.py'.")
    return chunks, np.asarray(embeddings, dtype=np.float32)


def retrieve(
    query: str,
    model: SentenceTransformer,
    chunks: list[Chunk],
    embeddings: np.ndarray,
) -> list[tuple[Chunk, float]]:
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )[0]
    similarities = embeddings @ query_embedding
    indices = np.argsort(similarities)[::-1][:TOP_K]
    return [(chunks[int(index)], float(similarities[index])) for index in indices]


def build_prompt(question: str, results: list[tuple[Chunk, float]]) -> str:
    best_score = results[0][1] if results else 0.0
    context = "\n\n".join(
        f"[{chunk.source}, chunk {chunk.chunk_number}, similarity {score:.3f}]\n{chunk.text}"
        for chunk, score in results
    )
    return f"""You are a careful cybersecurity information assistant.
Answer the question using only the supplied context. Do not invent facts, recommendations, or sources.
If the context does not contain enough evidence, say exactly that the supplied documents do not contain
enough relevant evidence to answer confidently, and briefly explain what is missing.
The highest retrieval similarity is {best_score:.3f}; treat scores below {MIN_RELEVANCE:.3f} as insufficient evidence.
Cite supporting source filenames in square brackets, such as [authentication.md].

Question:
{question}

Supplied context:
{context}
"""


def answer_question(question: str, results: list[tuple[Chunk, float]]) -> tuple[str, list[str]]:
    from openai import APIError, OpenAI

    sources = sorted({chunk.source for chunk, score in results if score >= MIN_RELEVANCE})
    if not results or not sources:
        return (
            "The supplied documents do not contain enough relevant evidence to answer confidently.",
            [],
        )

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required to generate an answer. Copy .env.example to .env and set it.")

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    models = [CHAT_MODEL] + [
        model.strip() for model in CHAT_MODEL_FALLBACKS.split(",") if model.strip()
    ]
    last_error: APIError | None = None
    for model in dict.fromkeys(models):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "Answer only from the retrieved cybersecurity context."},
                    {"role": "user", "content": build_prompt(question, results)},
                ],
            )
            break
        except APIError as error:
            last_error = error
    else:
        raise RuntimeError(f"All configured OpenRouter models failed: {last_error}") from last_error

    answer = response.choices[0].message.content or "The model returned an empty answer."
    return answer, sources


def run_query(question: str, model: SentenceTransformer, chunks: list[Chunk], embeddings: np.ndarray) -> None:
    results = retrieve(question, model, chunks, embeddings)
    answer, sources = answer_question(question, results)
    print(f"\nAnswer:\n{answer}")
    print("\nSource filenames:")
    print("\n".join(f"- {source}" for source in sources) or "- None: insufficient relevant evidence")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a small cybersecurity document RAG system.")
    parser.add_argument("question", nargs="*", help="Question to answer; omit for interactive mode")
    args = parser.parse_args()

    chunks, embeddings = load_embedding_cache()
    print(f"Loaded {len(chunks)} precomputed chunks from {len({chunk.source for chunk in chunks})} documents.")
    print(f"Loading query embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    if args.question:
        run_query(" ".join(args.question), model, chunks, embeddings)
        return

    while True:
        try:
            question = input("\nAsk a cybersecurity question (or 'exit' to quit): ").strip()
        except EOFError:
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            run_query(question, model, chunks, embeddings)


if __name__ == "__main__":
    main()