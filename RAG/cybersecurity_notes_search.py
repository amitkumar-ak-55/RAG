from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

NOTES = [
    "Phishing emails often impersonate trusted brands and push urgent action to steal credentials.",
    "Multi-factor authentication reduces account takeover risk even when passwords are exposed.",
    "Patch management is critical because unpatched software is a common entry point for ransomware.",
    "Endpoint detection and response tools help investigate suspicious process execution and lateral movement.",
    "Strong password policies combined with password managers reduce reuse and credential stuffing risk.",
    "Zero-trust architecture assumes no device or user is inherently trusted and verifies every request.",
    "Regular backups with offline copies help organizations recover from ransomware without paying extortion.",
    "Network segmentation limits the blast radius of an intrusion and slows attacker movement across systems.",
    "Security awareness training helps employees spot social engineering attempts and report suspicious requests.",
    "Least privilege access ensures users only receive the permissions required for their job duties.",
    "Security logs should be centralized and monitored to detect threats and support incident response.",
    "Application security testing, including code review and scanning, helps catch vulnerabilities before release.",
    "DNS filtering and web protection reduce exposure to malicious domains and phishing infrastructure.",
    "Incident response playbooks define how teams contain, eradicate, and recover from cyber incidents.",
    "Encryption protects sensitive data in transit and at rest, reducing the impact of data theft.",
    "Privileged accounts require tighter controls, session monitoring, and just-in-time access approvals."
]


def embed_notes(model: SentenceTransformer, notes: list[str]) -> np.ndarray:
    embeddings = model.encode(notes, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def rank_notes(query: str, notes: list[str], embeddings: np.ndarray, model: SentenceTransformer) -> list[tuple[int, float, str]]:
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    similarities = embeddings @ query_embedding
    ranked_indices = np.argsort(similarities)[::-1][:3]
    return [(int(index), float(similarities[index]), notes[index]) for index in ranked_indices]


def main() -> None:
    print("Loading model: sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer(MODEL_NAME)
    print("Generating embeddings for the cybersecurity notes...")
    embeddings = embed_notes(model, NOTES)

    while True:
        try:
            query = input("\nEnter a cybersecurity search query (or 'exit' to quit): ").strip()
        except EOFError:
            print("\nNo input received. Exiting.")
            return

        if not query:
            print("Please enter a query.")
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        ranked = rank_notes(query, NOTES, embeddings, model)

        print("\nTop 3 relevant notes:")
        for rank, (idx, score, note) in enumerate(ranked, start=1):
            print(f"{rank}. Score: {score:.4f} | Note #{idx + 1}")
            print(f"   {note}")


if __name__ == "__main__":
    main()
