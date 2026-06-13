"""Knowledge-base search tools backed by Redis (RediSearch).

kb_search_bm25: full-text BM25 search (OR-semantics keyword query).
kb_search_vector: HNSW vector search over gemini-embedding-001 embeddings
(available only when the index was built with embeddings).

Replies are parsed via execute_command so both the classic array reply and
the Redis 8 map-style reply work regardless of redis-py version."""

import os
import re
import struct

import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
KB_INDEX = "kb_idx"
DOC_PREFIX = "doc:"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
# Default number of BM25 results (env-overridable, read at import). BM25 is the
# primary retriever in the default CS_RETRIEVAL_MODE, so we return a few extra
# docs to improve recall; set KB_BM25_TOP_K=5 to restore the original default.
_DEFAULT_BM25_TOP_K = int(os.environ.get("KB_BM25_TOP_K", "8"))

_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
_genai_client = None


def _get_genai_client():
    """Reused genai client (one connection pool, not a new one per search)."""
    global _genai_client
    if _genai_client is None:
        from google import genai

        _genai_client = genai.Client()
    return _genai_client


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with gemini-embedding-001 via google-genai."""
    from google.genai import types

    # Reduced-dim output is unnormalized; the index uses COSINE, so that's fine.
    result = _get_genai_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    return [e.values for e in result.embeddings]


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _parse_search_reply(reply) -> list[dict]:
    """Normalize an FT.SEARCH reply (array or map shape) to result dicts."""
    if isinstance(reply, dict):
        results = reply.get(b"results", reply.get("results")) or []
        out = []
        for row in results:
            attrs = row.get(b"extra_attributes", row.get("extra_attributes")) or {}
            doc = {"doc_id": _decode(row.get(b"id", row.get("id", "")))}
            doc.update({_decode(k): _decode(v) for k, v in attrs.items()})
            out.append(doc)
        return out
    out = []
    for i in range(1, len(reply) - 1, 2):
        doc = {"doc_id": _decode(reply[i])}
        fields = reply[i + 1]
        for j in range(0, len(fields) - 1, 2):
            doc[_decode(fields[j])] = _decode(fields[j + 1])
        out.append(doc)
    return out


def _strip_score(docs: list[dict]) -> list[dict]:
    for doc in docs:
        doc.pop("score", None)
    return docs


def kb_search_bm25(query: str, top_k: int = _DEFAULT_BM25_TOP_K) -> list[dict]:
    """Full-text (BM25) search over the Rho-Bank knowledge base.

    Args:
        query: Keywords or a short phrase to search for. Matching is ranked,
            so extra keywords help rather than hurt.
        top_k: Number of documents to return.

    Returns:
        Matching documents with doc_id, title, and full content.
    """
    terms = re.findall(r"\w+", query.lower())
    if not terms:
        return []
    # OR-join: RediSearch defaults to AND, which zeroes out long queries.
    or_query = "|".join(dict.fromkeys(terms))
    reply = _client.execute_command(
        "FT.SEARCH", KB_INDEX, or_query,
        "LIMIT", "0", str(top_k),
        "RETURN", "2", "title", "content",
    )
    return _parse_search_reply(reply)


def kb_search_vector(query: str, top_k: int = 5) -> list[dict]:
    """Semantic (vector) search over the Rho-Bank knowledge base.

    Better than kb_search_bm25 when the query is a natural-language question
    rather than exact keywords.

    Args:
        query: A natural-language question or description.
        top_k: Number of documents to return.

    Returns:
        Matching documents with doc_id, title, and full content; or an error
        entry telling you to fall back to kb_search_bm25.
    """
    try:
        vector = struct.pack(f"{EMBEDDING_DIM}f", *_embed([query])[0])
        reply = _client.execute_command(
            "FT.SEARCH", KB_INDEX, f"*=>[KNN {top_k} @embedding $vec AS score]",
            "PARAMS", "2", "vec", vector,
            "SORTBY", "score",
            "LIMIT", "0", str(top_k),
            "RETURN", "3", "title", "content", "score",
            "DIALECT", "2",
        )
        return _strip_score(_parse_search_reply(reply))
    except Exception as e:
        return [
            {
                "error": f"Vector search unavailable ({type(e).__name__}). "
                "Use kb_search_bm25 with keywords instead."
            }
        ]
