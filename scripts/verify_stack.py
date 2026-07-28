"""verify_stack.py — e2e proof against a RUNNING stack (compose or Cloud Run).

Loads the sample document, waits for ready, asks a question over the WebSocket, and
asserts: node events streamed, a cited answer with structured claims arrived, and no
unverified figures. Exits non-zero on any failure.

    uv run python scripts/verify_stack.py [base_url]     (default http://localhost:8000)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx
import websockets


async def main(base: str) -> None:
    async with httpx.AsyncClient(base_url=base, timeout=30) as client:
        # /health, not /healthz: Cloud Run's queue-proxy swallows the latter
        health = (await client.get("/health")).json()
        print(f"health: {health}")

        doc = (await client.post("/load-sample")).json()
        assert "doc_id" in doc, f"load-sample failed: {doc}"
        doc_id = doc["doc_id"]
        t0 = time.time()
        while doc["status"] != "ready":
            assert doc["status"] != "error", f"ingest error: {doc.get('error')}"
            assert time.time() - t0 < 120, "ingest timed out"
            await asyncio.sleep(1)
            doc = (await client.get(f"/doc/{doc_id}/status")).json()
        print(f"doc ready: {doc_id} pages={doc['page_count']} chunks={doc['chunks']}")

        page = (await client.get(f"/doc/{doc_id}/page/4")).json()
        assert page["blocks"], "page blocks missing"

    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    async with websockets.connect(f"{ws_base}/ws/ask/{doc_id}") as ws:
        await ws.send(json.dumps({"question": "What was operating profit in FY26?"}))
        events = []
        while True:
            ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            events.append(ev)
            if ev["type"] in ("agent_done", "error"):
                break

    kinds = [e.get("node") or e["type"] for e in events]
    print(f"events: {kinds}")
    assert "error" not in [e["type"] for e in events], f"agent error: {events[-1]}"
    answer = next(e for e in events if e["type"] == "agent_answer")
    check = next(e for e in events if e.get("node") == "cite_check")
    assert answer["claims"], "no structured claims"
    assert answer["sources"], "no cited sources"
    assert check["unverified"] == [], f"unverified figures: {check['unverified']}"
    pages = sorted({c["page"] for cl in answer["claims"] for c in cl["citations"]})
    print(f"OK — cited answer with {len(answer['claims'])} claims (pages {pages}), "
          f"all figures verified")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))
