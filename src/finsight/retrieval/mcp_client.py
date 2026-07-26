"""mcp_client.py — the Retriever seam served over MCP (Streamable HTTP).

`MCPRetriever` satisfies the same `Retriever` protocol as `HybridRetriever`, so the
agent cannot tell whether retrieval runs in-process or in the sidecar — that symmetry
is the whole point of the seam. Stateless per call (the server runs stateless too):
open session → call `search_document` → parse Evidence JSON.
"""

from __future__ import annotations

import asyncio
import json

from .base import Evidence


class MCPRetriever:
    def __init__(self, url: str, doc_id: str | None = None):
        self.url = url
        self.doc_id = doc_id

    def retrieve(self, query: str, k: int = 6) -> list[Evidence]:
        return asyncio.run(self._call(query, k))

    async def _call(self, query: str, k: int) -> list[Evidence]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        args: dict = {"query": query, "k": k}
        if self.doc_id:
            args["doc_id"] = self.doc_id
        async with (streamablehttp_client(self.url) as (read, write, _),
                    ClientSession(read, write) as session):
            await session.initialize()
            result = await session.call_tool("search_document", args)
        payload = json.loads(result.content[0].text)
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(f"MCP retrieval failed: {payload['error']}")
        return payload
