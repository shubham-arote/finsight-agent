"""Public surface of the retrieval layer."""

from .base import Evidence, Retriever
from .embeddings import CohereEmbedder, get_embedder
from .hybrid import HybridRetriever, classify, lookup_terms, make_retriever
from .qdrant_index import QdrantIndex
from .text import sparse_counts, term_id, tok

__all__ = [
           "CohereEmbedder",
           "Evidence",
           "HybridRetriever",
           "QdrantIndex",
           "Retriever",
           "classify",
           "get_embedder",
           "lookup_terms",
           "make_retriever",
           "sparse_counts",
           "term_id",
           "tok",
]
