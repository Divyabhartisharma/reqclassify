import os
import math
import re
from collections import Counter
from supabase import create_client
from gemini_client import get_client

EMBED_MODEL = "models/gemini-embedding-001"


#def get_supabase():
 #   return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

def get_supabase():
    try:
        import streamlit as st
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")

    if not url or not url.startswith("https://"):
        raise RuntimeError(f"SUPABASE_URL invalid or missing: '{url}'")

    return create_client(url, key)

# simple in-memory cache to avoid re-embedding the same text
_EMBED_CACHE: dict = {}


def embed_text(text: str) -> list:
    key = text.strip()
    if key in _EMBED_CACHE:
        return _EMBED_CACHE[key]
    client = get_client()
    emb = client.models.embed_content(model=EMBED_MODEL, contents=key)
    vec = emb.embeddings[0].values
    _EMBED_CACHE[key] = vec
    return vec


def _tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_score(query: str, document: str, k1: float = 1.5, b: float = 0.75) -> float:
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(document)
    if not doc_tokens:
        return 0.0
    doc_len = len(doc_tokens)
    avg_doc_len = 100
    doc_freq = Counter(doc_tokens)
    score = 0.0
    for term in set(query_tokens):
        tf = doc_freq.get(term, 0)
        if tf == 0:
            continue
        idf = math.log(1 + (1 / (tf + 0.5)))
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
        score += idf * (numerator / denominator)
    return score


def retrieve_iso_context_by_vec(
    query_vec,
    standard_type: str,
    top_k: int = 3,
    fetch_k: int = 10,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    query_text: str = "",
) -> str:
    supabase = get_supabase()

    res = supabase.rpc(
        "match_iso_chunks",
        {
            "query_embedding": query_vec,
            "match_count": fetch_k,
            "standard_filter": standard_type,
        },
    ).execute()

    candidates = res.data or []
    if not candidates:
        return "No ISO context found."

    scored = []
    for item in candidates:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        cosine_sim = float(item.get("similarity", 0.5))
        bm25_norm = min(_bm25_score(query_text, content) / 5.0, 1.0) if query_text else 0.0
        combined = (vector_weight * cosine_sim) + (keyword_weight * bm25_norm)
        scored.append({"content": content, "combined": round(combined, 4)})

    scored.sort(key=lambda x: x["combined"], reverse=True)

    result_parts = []
    for i, chunk in enumerate(scored[:top_k]):
        result_parts.append(f"[ISO Chunk {i+1} | score={chunk['combined']}]\n{chunk['content']}")

    return "\n\n".join(result_parts)


def retrieve_example_context_by_vec(
    query_vec,
    top_k: int = 3,
    fetch_k: int = 10,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
    query_text: str = "",
    table_name: str = "requirement_examples",
) -> str:
    supabase = get_supabase()

    rpc_map = {
        "requirement_examples": "match_requirement_examples",
        "examples_autosar":     "match_examples_autosar",
        "examples_promise":     "match_examples_promise",
        "examples_userstory":   "match_examples_userstory",
        "examples_aerobert":    "match_examples_aerobert",
    }
    rpc_fn = rpc_map.get(table_name, "match_requirement_examples")

    res = supabase.rpc(
        rpc_fn,
        {
            "query_embedding": query_vec,
            "match_count": fetch_k,
        },
    ).execute()

    candidates = res.data or []
    if not candidates:
        return "No similar examples found."

    scored = []
    for item in candidates:
        req_text = (item.get("requirement") or "").strip()
        label = (item.get("label") or "").strip()
        if not req_text or not label:
            continue
        cosine_sim = float(item.get("similarity", 0.5))
        bm25_norm = min(_bm25_score(query_text, req_text) / 5.0, 1.0) if query_text else 0.0
        combined = (vector_weight * cosine_sim) + (keyword_weight * bm25_norm)
        scored.append({"requirement": req_text, "label": label, "combined": round(combined, 4)})

    scored.sort(key=lambda x: x["combined"], reverse=True)

    result_parts = []
    for i, ex in enumerate(scored[:top_k]):
        result_parts.append(
            f"[Example {i+1} | score={ex['combined']}]\n"
            f"Requirement: {ex['requirement']}\nLabel: {ex['label']}"
        )

    return "\n\n".join(result_parts)


def store_human_feedback(
    requirement: str,
    label: str,
    table_name: str = "requirement_examples"
) -> None:
    supabase = get_supabase()
    vec = embed_text(requirement)
    supabase.table(table_name).insert({
        "requirement": requirement,
        "label": label,
        "source": "human_confirmed",
        "embedding": vec,
    }).execute()