"""qdrant_index.py — the vector store: sparse (BM25-style) + optional dense, one collection.

Key-optional by construction:
  * no QDRANT_URL   -> in-process `QdrantClient(":memory:")` — dev/tests need no server
  * no COHERE key   -> sparse-only (IDF-weighted term matching); dense added when keyed
Phase 2 layers hybrid fusion + Cohere rerank + deterministic number lookup on top of
this; the collection schema already carries both vector types so no reindex is needed.
"""

from __future__ import annotations

import logging
import uuid

from qdrant_client import QdrantClient, models

from ..config import settings
from ..ingestion.chunking import Chunk
from .embeddings import CohereEmbedder, get_embedder
from .text import sparse_counts

logger = logging.getLogger(__name__)


class QdrantIndex:
    def __init__(self, client: QdrantClient | None = None, collection: str | None = None,
                 embedder: CohereEmbedder | str | None = "auto"):
        if client is None:
            client = (QdrantClient(url=settings.qdrant_url,
                                   api_key=settings.qdrant_api_key or None)
                      if settings.qdrant_url else QdrantClient(":memory:"))
        self.client = client
        self.collection = collection or settings.qdrant_collection
        self.embedder = get_embedder() if embedder == "auto" else embedder

    # ── schema ──────────────────────────────────────────────────────────────
    def ensure_collection(self) -> None:
        """Create the collection, or reconcile with one that already exists. A collection
        created keyless has no 'dense' named vector, and Qdrant can't add one later —
        so if a Cohere key appears afterwards we degrade to sparse-only writes/reads
        instead of 400-ing every upsert (recreate the collection to enable dense)."""
        if self.client.collection_exists(self.collection):
            if self.embedder:
                info = self.client.get_collection(self.collection)
                vectors = getattr(info.config.params, "vectors", None) or {}
                if "dense" not in vectors:
                    self.embedder = None       # schema predates the key: sparse-only
            return
        dense = {}
        if self.embedder:
            dense = {"dense": models.VectorParams(size=self.embedder.dim,
                                                  distance=models.Distance.COSINE)}
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=dense,
            sparse_vectors_config={"sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF)},
        )

    # ── write ───────────────────────────────────────────────────────────────
    def index_chunks(self, chunks: list[Chunk], batch: int = 128) -> int:
        self.ensure_collection()
        dense_vecs = None
        if self.embedder:
            try:
                dense_vecs = self.embedder.embed_docs([c.embed_text for c in chunks])
            except Exception as e:
                # Key-optional degradation applies to embeddings too: a dead/exhausted
                # embedding key must cost us dense retrieval, NOT the document. This
                # was failing the whole ingest when a trial quota ran out.
                logger.warning("dense embedding unavailable (%s: %s) — indexing "
                               "sparse-only", type(e).__name__, str(e)[:120])
                dense_vecs = None
                # Sticky: once embeddings fail, stay sparse-only for this index. Mixing
                # points that have a dense vector with points that don't corrupts the
                # collection (local mode raises a numpy broadcast error on the next
                # search) — consistency matters more than a few dense points.
                self.embedder = None
        points = []
        for i, c in enumerate(chunks):
            idxs, vals = sparse_counts(c.embed_text)
            vector: dict = {"sparse": models.SparseVector(indices=idxs, values=vals)}
            if dense_vecs:
                vector["dense"] = dense_vecs[i]
            points.append(models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.id)),
                vector=vector, payload=c.payload()))
        for i in range(0, len(points), batch):
            self.client.upsert(self.collection, points[i:i + batch])
        return len(points)

    # ── read ────────────────────────────────────────────────────────────────
    @staticmethod
    def _filter(doc_id: str | None):
        return (models.Filter(must=[models.FieldCondition(
                    key="doc_id", match=models.MatchValue(value=doc_id))])
                if doc_id else None)

    def _query(self, vector, using: str, n: int, doc_id: str | None) -> list[dict]:
        res = self.client.query_points(
            self.collection, query=vector, using=using,
            query_filter=self._filter(doc_id), limit=n, with_payload=True)
        return [{**p.payload, "score": p.score} for p in res.points]

    def sparse_candidates(self, query: str, n: int = 30, doc_id: str | None = None) -> list[dict]:
        idxs, vals = sparse_counts(query)
        if not idxs:
            return []
        return self._query(models.SparseVector(indices=idxs, values=vals), "sparse", n, doc_id)

    def dense_candidates(self, query: str, n: int = 30, doc_id: str | None = None) -> list[dict]:
        if not self.embedder:
            return []
        return self._query(self.embedder.embed_query(query), "dense", n, doc_id)

    def search(self, query: str, k: int = 6, doc_id: str | None = None) -> list[dict]:
        """Plain sparse search (CLI / debugging). The full stack is retrieval.hybrid."""
        return self.sparse_candidates(query, n=k, doc_id=doc_id)
