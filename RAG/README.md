# Cybersecurity RAG

This small retrieval-augmented generation example loads seven cybersecurity Markdown documents, splits each paragraph into a chunk, embeds chunks with the existing `sentence-transformers/all-MiniLM-L6-v2` model, retrieves the top three chunks, and asks an LLM to answer from that context. Embeddings are generated once and reused between queries.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`. The default chat model is `minimax/minimax-m3.1-flash:free`; if it fails, the app tries `nvidia/nemotron-3.5-super-vl:free` and `dots-studio/dots-3-note-preview:free`. You can change `CHAT_MODEL` and `CHAT_MODEL_FALLBACKS` for other OpenRouter models.

## Run

```powershell
python RAG\embed_documents.py
python rag.py "How can an organization reduce account takeover risk?"
python rag.py
```

Run `embed_documents.py` once after adding or editing documents. The query app reuses the saved `embeddings.npz` cache and tells you to rebuild if the documents or embedding model changed.

Every query retrieves exactly three chunks. The answer includes source filenames whose similarity meets `MIN_RELEVANCE`. When no retrieved chunk meets that threshold, the program reports that the documents do not contain enough relevant evidence and does not call the LLM.