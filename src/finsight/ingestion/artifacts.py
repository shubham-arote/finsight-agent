"""artifacts.py — SQLite cache for parse results and chunk contexts.

Why: cloud OCR and contextual-chunking calls cost rate-limited quota. Caching by
content hash makes ingestion idempotent and resumable — a rerun of the same PDF
re-parses nothing, and an OCR run interrupted at page 40 resumes at page 41.

    page_artifacts : (doc_hash, page, parser) -> blocks JSON
    kv             : content-hash key -> text (chunk-context cache)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from ..config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS page_artifacts (
    doc_hash TEXT NOT NULL,
    page     INTEGER NOT NULL,
    parser   TEXT NOT NULL,
    blocks   TEXT NOT NULL,
    PRIMARY KEY (doc_hash, page, parser)
);
CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def doc_hash(data: bytes) -> str:
    """Content identity of a document — same bytes, same cache entries."""
    return hashlib.sha256(data).hexdigest()[:16]


class ArtifactStore:
    def __init__(self, path: str | Path | None = None):
        p = str(path or settings.artifacts_db)
        if p != ":memory:":
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(p, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # ── page parses ─────────────────────────────────────────────────────────
    def get_page(self, doc: str, page: int, parser: str) -> list[dict] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT blocks FROM page_artifacts WHERE doc_hash=? AND page=? AND parser=?",
                (doc, page, parser)).fetchone()
        return json.loads(row[0]) if row else None

    def save_page(self, doc: str, page: int, parser: str, blocks: list[dict]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO page_artifacts VALUES (?,?,?,?)",
                (doc, page, parser, json.dumps(blocks, ensure_ascii=False)))
            self._conn.commit()

    def cached_pages(self, doc: str, parser: str) -> set[int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT page FROM page_artifacts WHERE doc_hash=? AND parser=?",
                (doc, parser)).fetchall()
        return {r[0] for r in rows}

    # ── generic text cache (chunk contexts) ─────────────────────────────────
    def get_text(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def save_text(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, value))
            self._conn.commit()
