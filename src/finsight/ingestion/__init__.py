"""Public surface of the ingestion layer."""

from .artifacts import ArtifactStore, doc_hash
from .chunking import Chunk, Section, build_chunks
from .models import BBox, Block, BlockType, block_from_dict, block_to_dict
from .pipeline import IngestError, IngestResult, ingest, parse_pdf

__all__ = ["ArtifactStore", "doc_hash", "Chunk", "Section", "build_chunks",
           "BBox", "Block", "BlockType", "block_from_dict", "block_to_dict",
           "IngestError", "IngestResult", "ingest", "parse_pdf"]
