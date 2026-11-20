# SmartQAAssistant_Analysis.md

Project root: c:\Users\Sheeba\GENAIPOC\SmartQAAssistant

Generated: 2025-10-13

---

## Summary
SmartQAAssistant is an AI assistant aimed at QA/business users to query test artifacts (test cases, defects, issues, requirements, user stories) and return concise answers, summaries, and links to source artifacts. Core capabilities implemented or expected in the repo:

- Vector search using FAISS
- Embeddings via local SentenceTransformers (preferred) with fallback to OpenAI embeddings
- Connectors for Jira and Excel-based test repositories
- LLM-driven summarization and business insights
- Web/UX surface (Streamlit / FastAPI expected)

This analysis covers files that are present and describes expected/missing files referenced in README and code.

---

## Files inspected

### README.md
- Purpose: project overview, setup notes, tools list, recommendations.
- Observations:
  - Contains high-level setup steps but formatting has minor issues (PowerShell snippet lines).
  - Recommends environment variables for secrets (good).
  - Documents main modules expected in repo.

### requirements.txt
- Contents (declared dependencies):
  - fastapi
  - uvicorn[standard]
  - openai
  - faiss-cpu
  - python-dotenv
  - pydantic
  - rich
  - numpy
  - tqdm
  - streamlit
- Observations:
  - No explicit versions; consider pinning to reproducible versions.
  - Dependencies reveal intended stack: FastAPI backend, FAISS vector DB, OpenAI for LLM/embeddings, and Streamlit for UI.

### src/ai_chatbot/db_connect.py
- File inspected (active file). Key points:
  - Purpose: load FAISS index and metadata, provide embed_text() and search() functions.
  - Tools/imports used: pickle, faiss, numpy, pathlib, sentence_transformers (local), openai (fallback).
  - Behavior:
    - Attempts to import SentenceTransformer and instantiate "all-MiniLM-L6-v2".
    - On import failure (MemoryError, OSError, ImportError) it falls back to OpenAI embeddings (creates client = OpenAI()).
    - Loads FAISS index at module import from db/faiss_index.bin.
    - Loads metadata pickle from db/index_metadata.pkl at module import.
    - embed_text(text): returns numpy float32 embedding from local model or OpenAI.
    - search(query, k): embeds query, does FAISS search, maps indices to metadata items and returns them.
- Risks & issues:
  - Heavy I/O and model loading happen at import time — slows startup and makes failures occur on import.
  - No existence checks or informative errors if index/metadata files missing or unreadable.
  - No dimensionality validation between embeddings and FAISS index.
  - OpenAI client initialization lacks explicit API key handling and error handling for network/API failures.
  - Uses a broad except catching MemoryError, OSError, ImportError only — other runtime errors may be swallowed or not handled.
- Recommended fix: lazy initialization, validation, clearer errors, enforce env var API key usage.
- Suggested replacement (safe, lazy-loading). Apply to src/ai_chatbot/db_connect.py:

````python
# filepath: c:\Users\Sheeba\GENAIPOC\SmartQAAssistant\src\ai_chatbot\db_connect.py
# ...existing code...
import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from typing import List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_PATH = PROJECT_ROOT / "db" / "faiss_index.bin"
META_PATH = PROJECT_ROOT / "db" / "index_metadata.pkl"

# Lazy-loaded globals
_model = None
_client = None
_index = None
_meta = None
_USE_OPENAI = None

def _init_embedder():
    global _model, _client, _USE_OPENAI
    if _model is not None or _client is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _USE_OPENAI = False
    except Exception:
        # Fallback to OpenAI
        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set in environment")
            _client = OpenAI(api_key=api_key)
            _USE_OPENAI = True
        except Exception as e:
            raise RuntimeError("No embedding provider available (SentenceTransformer or OpenAI).") from e

def _load_index_and_meta():
    global _index, _meta
    if _index is not None and _meta is not None:
        return
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(f"Index or metadata missing. Expected: {INDEX_PATH}, {META_PATH}")
    try:
        _index = faiss.read_index(str(INDEX_PATH))
    except Exception as e:
        raise RuntimeError(f"Failed to read FAISS index: {e}") from e
    with open(META_PATH, "rb") as f:
        _meta = pickle.load(f)

def embed_text(text: str) -> np.ndarray:
    """Embed text using local model or OpenAI (lazy init). Returns float32 1D array."""
    _init_embedder()
    if _USE_OPENAI is False:
        emb = _model.encode([text], convert_to_numpy=True)[0]
        return emb.astype(np.float32)
    else:
        resp = _client.embeddings.create(model="text-embedding-3-small", input=text)
        return np.array(resp.data[0].embedding, dtype=np.float32)

def search(query: str, k: int = 5) -> List[Any]:
    """Search FAISS for top-k matches. Returns list of metadata items."""
    _init_embedder()
    _load_index_and_meta()
    q_emb = embed_text(query).reshape(1, -1)
    # dimension check
    if q_emb.shape[1] != _index.d:
        raise RuntimeError(f"Embedding dim {q_emb.shape[1]} != index dim {_index.d}. Recreate index or use matching model.")
    D, I = _index.search(q_emb, k)
    results = []
    for idx in I[0]:
        if idx >= 0:
            try:
                results.append(_meta["items"][int(idx)])
            except Exception:
                continue
    return results
# ...existing code...