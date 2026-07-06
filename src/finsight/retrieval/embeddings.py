"""embeddings.py — dense embeddings via the Cohere API (LiteLLM), key-optional.

No key -> `get_embedder()` returns None and the index runs sparse-only (BM25-style).
"""

from __future__ import annotations

import litellm

from ..config import settings

_BATCH = 96


class CohereEmbedder:
    def __init__(self):
        self.model = settings.embed_model
        self.dim = settings.embed_dim

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), _BATCH):
            resp = litellm.embedding(model=self.model, input=texts[i:i + _BATCH],
                                     input_type=input_type,
                                     api_key=settings.cohere_api_key)
            out.extend(d["embedding"] for d in resp.data)
        return out

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "search_document")

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query], "search_query")[0]


def get_embedder() -> CohereEmbedder | None:
    return CohereEmbedder() if settings.cohere_api_key else None
