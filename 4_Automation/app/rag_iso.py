from gemini_client import get_client
from supabase import create_client
import os

EMBED_MODEL = "models/gemini-embedding-001"


def retrieve_iso_context(requirements, top_k=3, standard_type="iso25010"):
    """
    Retrieves ISO definition chunks (iso25010 or iso29148)
    using similarity search on ISO_Chunks.
    """

    client = get_client()

    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"],
    )

    # Combine batch into single embedding query
    combined_text = " ".join(requirements)

    emb = client.models.embed_content(
        model=EMBED_MODEL,
        contents=combined_text
    )

    query_vec = emb.embeddings[0].values

    # Call new RPC
    res = supabase.rpc(
        "match_iso_chunks",
        {
            "query_embedding": query_vec,
            "match_count": top_k,
            "standard_filter": standard_type,
        }
    ).execute()

    chunks = []
    for r in res.data or []:
        txt = (r.get("content") or "").strip()
        if txt:
            chunks.append(txt)

    return "\n\n".join(chunks)