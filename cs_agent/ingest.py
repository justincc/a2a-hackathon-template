"""Build the Redis knowledge-base index from kb/documents at startup.

Runs before the agent is served (main.py imports it), so the agent card only
becomes available once the index is ready. To avoid re-embedding on every
restart, an already-populated Redis index is reused as-is (the redis container
outlives cs-agent restarts); only a fresh/incomplete index is (re)built. When a
build does embed, results load from the pre-baked cache (kb/embeddings.json)
when present, else fall back to live embedding (and are written back to that
cache); without model credentials the index is BM25-only.

Embeddings are SKIPPED entirely (BM25-only, ~instant startup, no Vertex calls)
when KB_SKIP_EMBEDDINGS is set, or by default when CS_RETRIEVAL_MODE=bm25_primary
(the default) -- vector search is then only a last resort. Set
KB_SKIP_EMBEDDINGS=0 to force embedding, or KB_FORCE_REINDEX=1 to always rebuild.
Live embedding logs per-batch progress; tune batch size with KB_EMBED_BATCH_SIZE."""

import base64
import json
import os
import struct
import sys
from pathlib import Path

import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

from rag_tools import (
    DOC_PREFIX,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    KB_INDEX,
    REDIS_URL,
    _embed,
)

KB_DOCUMENTS_DIR = Path(os.environ.get("KB_DOCUMENTS_DIR", "/app/kb/documents"))
# Pre-baked {doc_id: base64(float32)} cache (see precompute_embeddings.py).
KB_EMBEDDINGS_PATH = Path(os.environ.get("KB_EMBEDDINGS_PATH", "/app/kb/embeddings.json"))

EMBED_BATCH_SIZE = int(os.environ.get("KB_EMBED_BATCH_SIZE", "25"))


def _should_skip_embeddings() -> bool:
    """Whether to build the index BM25-only (no Vertex embedding calls).

    KB_SKIP_EMBEDDINGS=1/true forces skipping; =0/false forces embedding;
    unset/"auto" (default) ties it to the retrieval mode: skip when
    CS_RETRIEVAL_MODE=bm25_primary (the default), embed in legacy mode.
    """
    val = os.environ.get("KB_SKIP_EMBEDDINGS", "auto").lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return os.environ.get("CS_RETRIEVAL_MODE", "bm25_primary").lower() == "bm25_primary"


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


def _index_is_complete(
    client: redis.Redis, documents: list[dict], *, require_embedding: bool
) -> bool:
    """True if Redis already holds every document, so a restart can reuse it.

    With require_embedding=True every doc must also carry an 'embedding' field
    (so we never reuse a partial or BM25-only index when embeddings are wanted).
    With require_embedding=False only the doc hashes need to exist (used when
    embeddings are skipped). Any missing doc forces a rebuild.
    """
    try:
        client.ft(KB_INDEX).info()
    except redis.ResponseError:
        return False  # index does not exist yet
    pipe = client.pipeline(transaction=False)
    for doc in documents:
        key = f"{DOC_PREFIX}{doc['id']}"
        if require_embedding:
            pipe.hexists(key, "embedding")
        else:
            pipe.exists(key)
    return all(pipe.execute())


def _embed_misses(
    documents: list[dict], embedding_bytes: list[bytes | None], misses: list[int]
) -> None:
    """Live-embed the cache-miss documents in batches, logging per-batch progress.

    Embedding via Vertex is slow (~10s+/call), so a flushed progress line is
    emitted after every batch -- otherwise the multi-minute embed looks like a
    hang. Lower KB_EMBED_BATCH_SIZE for finer-grained progress (more API calls).
    Failures leave the affected docs BM25-only rather than aborting the build.
    """
    total = len(misses)
    print(
        f"[ingest] embedding {total} documents via {EMBEDDING_MODEL} "
        f"in batches of {EMBED_BATCH_SIZE}...",
        file=sys.stderr,
        flush=True,
    )
    try:
        for start in range(0, total, EMBED_BATCH_SIZE):
            idx = misses[start : start + EMBED_BATCH_SIZE]
            vectors = _embed(
                [f"{documents[i]['title']}\n{documents[i]['content']}" for i in idx]
            )
            for i, vector in zip(idx, vectors):
                embedding_bytes[i] = struct.pack(f"{EMBEDDING_DIM}f", *vector)
            done = min(start + EMBED_BATCH_SIZE, total)
            print(
                f"[ingest] embedded {done}/{total} documents ({done * 100 // total}%)",
                file=sys.stderr,
                flush=True,
            )
    except Exception as e:
        done = sum(1 for i in misses if embedding_bytes[i] is not None)
        print(
            f"[ingest] embeddings unavailable ({e}); {done}/{total} embedded before "
            "failure -- the rest will be BM25-only (kb_search_bm25 still works)",
            file=sys.stderr,
            flush=True,
        )


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
        print(f"[ingest] wrote {len(have)} embeddings to {KB_EMBEDDINGS_PATH}", file=sys.stderr, flush=True)
    except OSError as e:
        print(f"[ingest] could not persist embedding cache ({e})", file=sys.stderr, flush=True)


def build_index() -> None:
    """(Re)create the KB index and load every document, embedding if possible.

    Reuses an already-populated Redis index (no rebuild on restart). Embeddings
    are skipped when KB_SKIP_EMBEDDINGS is set or CS_RETRIEVAL_MODE=bm25_primary
    (the default), giving a BM25-only, ~instant startup. Set KB_FORCE_REINDEX=1
    to force a full rebuild (e.g. after editing documents).
    """
    client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No KB documents found in {KB_DOCUMENTS_DIR}")

    force = os.environ.get("KB_FORCE_REINDEX", "").lower() in ("1", "true", "yes")
    skip_embeddings = _should_skip_embeddings()

    if not force and _index_is_complete(
        client, documents, require_embedding=not skip_embeddings
    ):
        kind = "BM25-only" if skip_embeddings else "embedded"
        print(
            f"[ingest] reusing existing Redis index: {len(documents)} documents "
            f"({kind}); skipping rebuild (KB_FORCE_REINDEX=1 to rebuild)",
            file=sys.stderr,
            flush=True,
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

    embedding_bytes: list[bytes | None] = [None] * len(documents)
    misses: list[int] = []
    if skip_embeddings:
        print(
            f"[ingest] embeddings skipped (BM25-only); indexing {len(documents)} "
            "documents -- set KB_SKIP_EMBEDDINGS=0 to embed",
            file=sys.stderr,
            flush=True,
        )
    else:
        # Pre-baked cache first; live-embed only the misses (BM25-only if neither).
        cache = load_embedding_cache()
        embedding_bytes = [cache.get(d["id"]) for d in documents]
        misses = [i for i, b in enumerate(embedding_bytes) if b is None]
        if cache:
            print(
                f"[ingest] embedding cache hit for {len(documents) - len(misses)}/"
                f"{len(documents)} documents",
                file=sys.stderr,
                flush=True,
            )
        if misses:
            _embed_misses(documents, embedding_bytes, misses)

    pipe = client.pipeline(transaction=False)
    for doc, emb in zip(documents, embedding_bytes):
        mapping = {"title": doc["title"], "content": doc["content"]}
        if emb is not None:
            mapping["embedding"] = emb
        pipe.hset(f"{DOC_PREFIX}{doc['id']}", mapping=mapping)
    pipe.execute()
    print(f"[ingest] indexed {len(documents)} documents into {KB_INDEX}", file=sys.stderr, flush=True)

    # Save newly computed embeddings so the next cold start can skip embedding.
    if misses:
        _persist_embedding_cache(documents, embedding_bytes)


if __name__ == "__main__":
    build_index()
