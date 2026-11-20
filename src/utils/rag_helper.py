# src/utils/rag_helper.py
# Enhanced with token-aware RAG (Feature 2)

import chromadb
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path
from src.llm.llm_interface import LLMInterface, TokenUsage

# ── Flexible ChromaDB path resolution ──
def get_chroma_db_path():
    possible_paths = [
        "db/chroma_db",
        "./db/chroma_db",
        "../db/chroma_db",
        "../../db/chroma_db",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return "db/chroma_db"

CHROMA_DB_PATH = get_chroma_db_path()
COLLECTION_NAME = "defects_kedb"

_client = None
_collection = None
_llm = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = _client.get_collection(COLLECTION_NAME)
    return _collection


def get_llm():
    global _llm
    if _llm is None:
        _llm = LLMInterface(model_choice="llama3")
    return _llm


def retrieve_kedb_context(query: str, k: int = 5, where: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Retrieve defect documents from ChromaDB with optional metadata filtering."""
    collection = get_collection()
    query_params = {"query_texts": [query], "n_results": k}
    if where:
        query_params["where"] = where

    results = collection.query(**query_params)

    docs: List[Dict[str, Any]] = []
    if results and results.get("documents"):
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            docs.append({"page_content": doc, "metadata": meta})
    return docs


def _build_rag_prompt(user_query: str, docs: List[Dict[str, Any]]) -> str:
    """Build the full RAG prompt from context documents."""
    context_blocks = []
    for d in docs:
        did = d["metadata"].get("Issue key", "Unknown")
        context_blocks.append(f"[Issue key: {did}]\n{d['page_content']}")

    context_text = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a QA assistant chatbot for a software organization. "
        "You answer questions about defects, root cause, resolution, workarounds, test cases, "
        "test sets, and releases. Use ONLY the provided KEDB context. "
        "If the answer is not clearly covered, say you are not sure and suggest consulting the QA team."
    )

    user_part = (
        f"KEDB Context:\n{context_text}\n\n"
        f"User question: {user_query}\n\n"
        "Answer concisely based only on the KEDB context above. "
        "If there is not enough information, say you do not have enough information."
    )

    return system_prompt + "\n\n" + user_part


def answer_with_kedb_rag(user_query: str) -> str:
    """Original RAG answer function – preserved for backward compatibility."""
    llm = get_llm()
    docs = retrieve_kedb_context(user_query, k=5)
    if not docs:
        return "I could not find relevant information in the KEDB for this query."
    full_prompt = _build_rag_prompt(user_query, docs)
    return llm.generate(full_prompt)


def answer_with_kedb_rag_with_tokens(user_query: str) -> Tuple[str, TokenUsage]:
    """
    RAG answer function that also returns token usage (Feature 2).
    Returns:
        (answer_text, TokenUsage)
    """
    llm = get_llm()
    docs = retrieve_kedb_context(user_query, k=5)
    if not docs:
        return "I could not find relevant information in the KEDB for this query.", TokenUsage()
    full_prompt = _build_rag_prompt(user_query, docs)
    return llm.generate_with_tokens(full_prompt)
