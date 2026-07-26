"""Public surface of the ingestion layer."""

from .artifacts import ArtifactStore, doc_hash
from .chunking import Chunk, Section, build_chunks
from .models import BBox, Block, BlockType, block_from_dict, block_to_dict
from .pipeline import IngestError, IngestResult, ingest, parse_pdf

__all__ = [
           "ArtifactStore",
           "BBox",
           "Block",
           "BlockType",
           "Chunk",
           "IngestError",
           "IngestResult",
           "Section",
           "block_from_dict",
           "block_to_dict",
           "build_chunks",
           "doc_hash",
           "ingest",
           "parse_pdf",
]
