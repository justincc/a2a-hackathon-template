"""Build the Redis knowledge-base index from kb/documents at startup.

Runs before the agent is served (main.py imports it), so the agent card only
becomes available once the index is ready. To avoid re-embedding on every
restart, an already-populated Redis index is reused as-is (the redis container
outlives cs-agent restarts); only a fresh/incomplete index is (re)built. When a
build does embed, results load from the pre-baked cache (kb/embeddings.json)
when present, else fall back to live embedding (and are written back to that
cache); without model credentials the index is BM25-only. Set KB_FORCE_REINDEX=1
to always rebuild."""

import base64
import json
import os
import struct
import sys
from pathlib import Path

import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

from rag_tools import DOC_PREFIX, EMBEDDING_DIM, KB_INDEX, REDIS_URL, _embed

KB_DOCUMENTS_DIR = Path(os.environ.get("KB_DOCUMENTS_DIR", "/app/kb/documents"))
# Pre-baked {doc_id: base64(float32)} cache (see precompute_embeddings.py).
KB_EMBEDDINGS_PATH = Path(os.environ.get("KB_EMBEDDINGS_PATH", "/app/kb/embeddings.json"))

EMBED_BATCH_SIZE = 25


def load_embedding_cache() -> dict[str, bytes]:
    """Load pre-baked embedding bytes by doc id (empty dict if no cache)."""
    if not KB_EMBEDDINGS_PATH.exists():
        return {}
    with open(KB_EMBEDDINGS_PATH) as fp:
        raw = json.load(fp)
    return {doc_id: base64.b64decode(b64) for doc_id, b64 in raw.items()}


def load_documents() -> list[dict]:
    """Load all KB documents ({id, title, content})."""
    docs = []
    for path in sorted(KB_DOCUMENTS_DIR.glob("*.json")):
        with open(path) as fp:
            docs.append(json.load(fp))
    return docs


def _index_has_all_embeddings(client: redis.Redis, documents: list[dict]) -> bool:
    """True if Redis already holds every document WITH an embedding.

    Lets a restart reuse the embeddings already in Redis instead of
    re-embedding from scratch. A doc whose hash or 'embedding' field is missing
    forces a rebuild, so this returns False for an empty, partial, or
    BM25-only index.
    """
    try:
        client.ft(KB_INDEX).info()
    except redis.ResponseError:
        return False  # index does not exist yet
    pipe = client.pipeline(transaction=False)
    for doc in documents:
        pipe.hexists(f"{DOC_PREFIX}{doc['id']}", "embedding")
    return all(pipe.execute())


def _persist_embedding_cache(documents: list[dict], embedding_bytes: list[bytes | None]) -> None:
    """Best-effort: write computed embeddings back to KB_EMBEDDINGS_PATH so a
    future cold start (fresh Redis) can load them instead of re-embedding."""
    have = {
        doc["id"]: base64.b64encode(emb).decode()
        for doc, emb in zip(documents, embedding_bytes)
        if emb is not None
    }
    if not have:
        return
    try:
        KB_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        KB_EMBEDDINGS_PATH.write_text(json.dumps(have))
        print(f"[ingest] wrote {len(have)} embeddings to {KB_EMBEDDINGS_PATH}", file=sys.stderr)
    except OSError as e:
        print(f"[ingest] could not persist embedding cache ({e})", file=sys.stderr)


def build_index() -> None:
    """(Re)create the KB index and load every document, embedding if possible.

    Reuses an already-populated Redis index when every document is present with
    an embedding (no re-embedding on restart). Set KB_FORCE_REINDEX=1 to force a
    full rebuild (e.g. after editing documents).
    """
    client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No KB documents found in {KB_DOCUMENTS_DIR}")

    force = os.environ.get("KB_FORCE_REINDEX", "").lower() in ("1", "true", "yes")
    if not force and _index_has_all_embeddings(client, documents):
        print(
            f"[ingest] reusing existing Redis index: {len(documents)} documents "
            "already embedded (set KB_FORCE_REINDEX=1 to rebuild)",
            file=sys.stderr,
        )
        return

    try:
        client.ft(KB_INDEX).dropindex(delete_documents=True)
    except redis.ResponseError:
        pass

    client.ft(KB_INDEX).create_index(
        fields=[
            TextField("title", weight=2.0),
            TextField("content"),
            VectorField(
                "embedding",
                "HNSW",
                {"TYPE": "FLOAT32", "DIM": EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
            ),
        ],
        definition=IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH),
    )

    # Pre-baked cache first; live-embed only the misses (BM25-only if neither).
    cache = load_embedding_cache()
    embedding_bytes: list[bytes | None] = [cache.get(d["id"]) for d in documents]
    misses = [i for i, b in enumerate(embedding_bytes) if b is None]
    if cache:
        print(
            f"[ingest] embedding cache hit for {len(documents) - len(misses)}/"
            f"{len(documents)} documents",
            file=sys.stderr,
        )
    if misses:
        try:
            for start in range(0, len(misses), EMBED_BATCH_SIZE):
                idx = misses[start : start + EMBED_BATCH_SIZE]
                vectors = _embed([f"{documents[i]['title']}\n{documents[i]['content']}" for i in idx])
                for i, vector in zip(idx, vectors):
                    embedding_bytes[i] = struct.pack(f"{EMBEDDING_DIM}f", *vector)
            print(f"[ingest] live-embedded {len(misses)} uncached documents", file=sys.stderr)
        except Exception as e:
            print(
                f"[ingest] embeddings unavailable ({e}); {len(misses)} doc(s) "
                "will be BM25-only (kb_search_bm25 still works)",
                file=sys.stderr,
            )

    pipe = client.pipeline(transaction=False)
    for doc, emb in zip(documents, embedding_bytes):
        mapping = {"title": doc["title"], "content": doc["content"]}
        if emb is not None:
            mapping["embedding"] = emb
        pipe.hset(f"{DOC_PREFIX}{doc['id']}", mapping=mapping)
    pipe.execute()
    print(f"[ingest] indexed {len(documents)} documents into {KB_INDEX}", file=sys.stderr)

    # Save newly computed embeddings so the next cold start can skip embedding.
    if misses:
        _persist_embedding_cache(documents, embedding_bytes)


if __name__ == "__main__":
    build_index()
