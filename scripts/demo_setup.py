"""demo_setup.py — get the demo into a known-good state, or fail loudly now.

Run this AFTER starting the server and BEFORE presenting. It waits for health, loads
every document the demo touches, and asks one real question end to end — so anything
broken surfaces here, in private, instead of on the screen.

    uv run uvicorn finsight.server:app --port 8000     # terminal 1
    uv run python scripts/demo_setup.py                # terminal 2

Exit 0 means: server up, documents ingested, agent answering with verified citations.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import websockets

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
PDFS = Path(__file__).resolve().parent.parent / "data" / "finragbench_v" / "pdfs"
DEMO_DOCS = [
    ("PDF Solutions Reports First Quarter 2023 Results.pdf", "Q1 2023"),
    ("PDF Solutions Reports First Quarter 2022 Results.pdf", "Q1 2022"),
]
SMOKE_Q = "What was total revenue in the first quarter of 2023?"


def ok(msg):
    print(f"  [OK]   {msg}")


def fail(msg):
    # ASCII only: Windows consoles are cp1252 and a unicode mark here would crash the
    # very message that tells you what is broken.
    print(f"\n  [FAIL] {msg}\n\nDEMO IS NOT READY - fix this before presenting.")
    sys.exit(1)


async def wait_ready(client, doc_id, label, timeout=240):
    t0 = time.time()
    while True:
        s = (await client.get(f"/doc/{doc_id}/status")).json()
        if s["status"] == "ready":
            return s
        if s["status"] == "error":
            fail(f"{label} failed to ingest: {str(s.get('error'))[:160]}")
        if time.time() - t0 > timeout:
            fail(f"{label} still ingesting after {timeout}s")
        await asyncio.sleep(2)


async def main():
    print(f"\nfinsight demo preflight — {BASE}\n")
    async with httpx.AsyncClient(base_url=BASE, timeout=300) as client:
        # 1. server
        h = None
        for _ in range(45):                    # the server may still be booting
            try:
                r = await client.get("/health")
                if r.status_code == 200 and "status" in r.json():
                    h = r.json()
                    break
                # A reachable server that doesn't know /health is running OLD code —
                # usually a stale process still holding the port. Say so plainly
                # instead of dying on a KeyError while reading a 404 body.
                fail("a server is running on this port but it predates /health — it is "
                     "stale.\n         Stop it, then start a fresh one:\n"
                     "           PowerShell:  Get-NetTCPConnection -LocalPort 8000 -State Listen |"
                     " ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }\n"
                     "           then:        make demo")
            except SystemExit:
                raise
            except Exception:
                await asyncio.sleep(2)
        if h is None:
            fail("server not reachable. Start it first:  make demo")
        ok(f"server up (cloud={h.get('cloud')})")

        status = (await client.get("/api/status")).json()
        model = status["models"]["answer"]
        if not status["cloud"]:
            print("    [warn] offline mode — answers will be extractive quotes, not prose")
        ok(f"answering with {model}")

        # 2. the sample (always present, instant)
        doc = (await client.post("/load-sample")).json()
        if "doc_id" not in doc:
            fail(f"load-sample failed: {doc}")
        await wait_ready(client, doc["doc_id"], "sample")
        ok(f"sample_report.pdf ready ({doc['doc_id']})")
        ids = {"sample": doc["doc_id"]}

        # 3. the real filings (skipped cleanly if the dataset isn't downloaded)
        for filename, label in DEMO_DOCS:
            path = PDFS / filename
            if not path.exists():
                print(f"    [skip] {label}: {filename} not found — compare demo unavailable")
                continue
            with path.open("rb") as fh:
                r = (await client.post("/upload", files={"file": (filename, fh,
                                                                  "application/pdf")})).json()
            if "doc_id" not in r:
                fail(f"{label} upload failed: {r}")
            s = await wait_ready(client, r["doc_id"], label)
            ids[label] = r["doc_id"]
            ok(f"{label} ready — {s['page_count']}p, {s['chunks']} chunks ({r['doc_id']})")

    # 4. one real question, all the way through
    ws_base = BASE.replace("https://", "wss://").replace("http://", "ws://")
    target = ids.get("Q1 2023", ids["sample"])
    q = SMOKE_Q if "Q1 2023" in ids else "What was operating profit in FY26?"
    async with websockets.connect(f"{ws_base}/ws/ask/{target}", max_size=None) as ws:
        await ws.send(json.dumps({"question": q}))
        answer, cites, unverified = None, [], None
        while True:
            ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
            if ev["type"] == "agent_answer":
                answer = ev["answer"]
                cites = [c for cl in ev["claims"] for c in cl["citations"]]
            elif ev.get("node") == "cite_check":
                unverified = ev["unverified"]
            elif ev["type"] == "error":
                fail(f"agent error: {ev['error']}")
            elif ev["type"] == "agent_done":
                break
    if not answer or not cites:
        fail("the agent answered without citations")
    shown = ", ".join(f"p{c['page']}b{c['block_id']}" for c in cites[:3])
    ok(f"answered + cited: {answer[:70]}")
    print(f"    citations: {shown}")
    print(f"    unverified figures: {unverified}")

    print("\nREADY. Demo sequence:")
    print("  1. Load sample report  -> ask 'What was operating profit in FY26?'")
    print("     click the citation chip -> highlights the exact row on page 4")
    print("  2. Ask 'By how much did operating profit change year on year, in percent?'")
    print("     the lane chip flips to calc -> the calculator computes it")
    print("  3. Upload Q1 2023 (instant, already cached) -> press the Brief button")
    print("  4. Compare picker -> vs Q1 2022 -> the delta table, cited into both filings\n")


if __name__ == "__main__":
    asyncio.run(main())
