from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer


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
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
RERANK_CANDIDATES = 10
TOP_K = 3
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-1.0"))


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
    reranker: CrossEncoder,
    chunks: list[Chunk],
    embeddings: np.ndarray,
) -> list[tuple[Chunk, float]]:
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )[0]
    similarities = embeddings @ query_embedding
    candidate_indices = np.argsort(similarities)[::-1][:RERANK_CANDIDATES]
    candidates = [chunks[int(index)] for index in candidate_indices]
    rerank_scores = np.asarray(
        reranker.predict([(query, chunk.text) for chunk in candidates]),
        dtype=np.float32,
    )
    ranked_candidates = sorted(
        zip(candidates, rerank_scores),
        key=lambda result: float(result[1]),
        reverse=True,
    )
    return [
        (chunk, float(score))
        for chunk, score in ranked_candidates[:TOP_K]
    ]


def build_prompt(question: str, results: list[tuple[Chunk, float]]) -> str:
    best_score = results[0][1] if results else 0.0
    context = "\n\n".join(
        f"[{chunk.source}, chunk {chunk.chunk_number}, reranker score {score:.3f}]\n{chunk.text}"
        for chunk, score in results
    )
    return f"""You are a careful cybersecurity information assistant.
Answer the question using only the supplied context. Do not invent facts, recommendations, or sources.
If the context does not contain enough evidence, say exactly that the supplied documents do not contain
enough relevant evidence to answer confidently, and briefly explain what is missing.
The highest reranker score is {best_score:.3f}; treat scores below {MIN_RERANK_SCORE:.3f} as insufficient evidence.
Cite supporting source filenames in square brackets, such as [authentication.md].

Question:
{question}

Supplied context:
{context}
"""


def answer_question(question: str, results: list[tuple[Chunk, float]]) -> tuple[str, list[str]]:
    from openai import APIError, OpenAI

    sources = sorted({chunk.source for chunk, score in results if score >= MIN_RERANK_SCORE})
    if not results or not sources:
        return (
            "The supplied documents do not contain enough relevant evidence to answer confidently.",
            [],
        )

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required to generate an answer. Copy .env.example to .env and set it.")

    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=45.0,
        max_retries=0,
    )
    models = [CHAT_MODEL] + [
        model.strip() for model in CHAT_MODEL_FALLBACKS.split(",") if model.strip()
    ]
    last_error: APIError | None = None
    for model in dict.fromkeys(models):
        try:
            print(f"Generating answer with {model} (timeout: 45s)...", flush=True)
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


def run_query(
    question: str,
    model: SentenceTransformer,
    reranker: CrossEncoder,
    chunks: list[Chunk],
    embeddings: np.ndarray,
) -> None:
    print("Retrieving relevant document context...", flush=True)
    results = retrieve(question, model, reranker, chunks, embeddings)
    print("Retrieved context. Sending it to OpenRouter...", flush=True)
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
    print(f"Loading reranker model: {RERANKER_MODEL}")
    reranker = CrossEncoder(RERANKER_MODEL)

    if args.question:
        run_query(" ".join(args.question), model, reranker, chunks, embeddings)
        return

    while True:
        try:
            question = input("\nAsk a cybersecurity question (or 'exit' to quit): ").strip()
        except EOFError:
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            run_query(question, model, reranker, chunks, embeddings)


if __name__ == "__main__":
    main()