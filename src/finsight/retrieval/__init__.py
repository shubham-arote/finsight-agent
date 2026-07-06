"""Public surface of the retrieval layer."""

from .base import Evidence, Retriever
from .embeddings import CohereEmbedder, get_embedder
from .hybrid import HybridRetriever, classify, lookup_terms, make_retriever
from .qdrant_index import QdrantIndex
from .text import sparse_counts, term_id, tok

__all__ = ["Evidence", "Retriever", "CohereEmbedder", "get_embedder",
           "HybridRetriever", "classify", "lookup_terms", "make_retriever",
           "QdrantIndex", "sparse_counts", "term_id", "tok"]
