"""models.py — the parsing data model: Block + reading order.

Ported from financial_analyst_agent `srr/core.py`, trimmed to what ingestion needs.
Every parser (text-layer, cloud OCR) emits the same `Block` objects, so chunking,
citations (page + bbox), and persistence work identically regardless of source.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class BlockType(str, Enum):
    TEXT = "text"
    TITLE = "title"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"


# "Furniture" — page chrome excluded from the reading flow and the index.
FURNITURE = {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER}


@dataclass
class BBox:
    """Bounding box in page coordinate space, origin top-left."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Block:
    type: BlockType
    bbox: BBox
    page: int = 0
    id: int | None = None           # per-page block index (citation anchor)
    content: str | None = None
    order: int | None = None        # reading order; None = furniture / excluded


def block_to_dict(b: Block) -> dict:
    # bbox persisted as exact floats (JSON round-trips them losslessly): any rounding made
    # cached blocks differ from fresh ones at furniture-band boundaries, breaking
    # parse determinism
    return {"id": b.id, "type": b.type.value,
            "bbox": [b.bbox.x0, b.bbox.y0, b.bbox.x1, b.bbox.y1],
            "page": b.page, "order": b.order, "content": b.content}


def block_from_dict(d: dict) -> Block:
    b = Block(type=BlockType(d["type"]), bbox=BBox(*d["bbox"]), page=int(d.get("page", 0)))
    b.id, b.order, b.content = d.get("id"), d.get("order"), d.get("content")
    return b


class ColumnAwareReadingOrder:
    """Reading order via recursive XY-cut over blocks: split by the widest whitespace gap —
    horizontal first (top-to-bottom bands), then vertical (left-to-right columns) — and
    recurse. Robust on Manhattan layouts (reports, filings)."""

    def __init__(self, min_gap: int = 14):
        self.min_gap = min_gap

    def _gap_split(self, blocks: list[Block], axis: str) -> list[list[Block]]:
        if axis == "y":
            lo, hi = (lambda b: b.bbox.y0), (lambda b: b.bbox.y1)
        else:
            lo, hi = (lambda b: b.bbox.x0), (lambda b: b.bbox.x1)
        sb = sorted(blocks, key=lo)
        groups: list[list[Block]] = [[sb[0]]]
        cur_end = hi(sb[0])
        for b in sb[1:]:
            if lo(b) - cur_end > self.min_gap:
                groups.append([b])
                cur_end = hi(b)
            else:
                groups[-1].append(b)
                cur_end = max(cur_end, hi(b))
        return groups

    def _cut(self, blocks: list[Block], out: list[Block]) -> None:
        if len(blocks) <= 1:
            out.extend(blocks)
            return
        rows = self._gap_split(blocks, "y")
        if len(rows) > 1:
            for g in rows:
                self._cut(g, out)
            return
        cols = self._gap_split(blocks, "x")
        if len(cols) > 1:
            for g in cols:
                self._cut(g, out)
            return
        out.extend(sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.x0)))

    def order(self, blocks: Sequence[Block], page_size: tuple[float, float]) -> list[Block]:
        body = [b for b in blocks if b.type not in FURNITURE]
        ordered: list[Block] = []
        if body:
            self._cut(list(body), ordered)
        for i, b in enumerate(ordered):
            b.order = i
        return ordered
