"""qdrant_index.py — the vector store: sparse (BM25-style) + optional dense, one collection.

Key-optional by construction:
  * no QDRANT_URL   -> in-process `QdrantClient(":memory:")` — dev/tests need no server
  * no COHERE key   -> sparse-only (IDF-weighted term matching); dense added when keyed
Phase 2 layers hybrid fusion + Cohere rerank + deterministic number lookup on top of
this; the collection schema already carries both vector types so no reindex is needed.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from ..config import settings
from ..ingestion.chunking import Chunk
from .embeddings import CohereEmbedder, get_embedder
from .text import sparse_counts


class QdrantIndex:
    def __init__(self, client: QdrantClient | None = None, collection: str | None = None,
                 embedder: CohereEmbedder | None | str = "auto"):
        if client is None:
            client = (QdrantClient(url=settings.qdrant_url,
                                   api_key=settings.qdrant_api_key or None)
                      if settings.qdrant_url else QdrantClient(":memory:"))
        self.client = client
        self.collection = collection or settings.qdrant_collection
        self.embedder = get_embedder() if embedder == "auto" else embedder

    # ── schema ──────────────────────────────────────────────────────────────
    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
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
        dense_vecs = (self.embedder.embed_docs([c.embed_text for c in chunks])
                      if self.embedder else None)
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

    # ── read (Phase 1: sparse-first; hybrid fusion arrives in Phase 2) ──────
    def search(self, query: str, k: int = 6, doc_id: str | None = None) -> list[dict]:
        flt = (models.Filter(must=[models.FieldCondition(
                   key="doc_id", match=models.MatchValue(value=doc_id))])
               if doc_id else None)
        idxs, vals = sparse_counts(query)
        res = self.client.query_points(
            self.collection,
            query=models.SparseVector(indices=idxs, values=vals),
            using="sparse", query_filter=flt, limit=k, with_payload=True)
        return [{**p.payload, "score": p.score} for p in res.points]
