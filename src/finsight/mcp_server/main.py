"""mcp_server — retrieval exposed as an MCP tool (week-4 sidecar pattern).

A Streamable-HTTP MCP server (Starlette + StreamableHTTPSessionManager, stateless)
wrapping the SAME retrieval stack the agent uses in-process: hybrid RRF + rerank +
exact lookup over Qdrant. One tool:

    search_document(query, doc_id?, k?) -> Evidence JSON (page/block/bbox/parent/...)

Deployed as a sidecar next to the agent container (shared localhost / compose network);
the agent switches to it with MCP_SERVER_URL — and falls back in-process without it,
so the key-optional/local story is unchanged.

Run:  uv run python -m finsight.mcp_server.main   (port: MCP_PORT, default 3000)
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncIterator

import mcp.types as types
import uvicorn
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount

from ..config import settings
from ..retrieval import HybridRetriever, QdrantIndex

logger = logging.getLogger(__name__)

INDEX = QdrantIndex()          # QDRANT_URL (compose/cloud) or in-process for tests

mcp_server = Server("finsight-retrieval")


@mcp_server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [types.Tool(
        name="search_document",
        description=("Hybrid search over ingested financial documents. Returns evidence "
                     "chunks with citation anchors (page, block_id, bbox), the matched "
                     "text, and the parent section context."),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "doc_id": {"type": "string", "description": "scope to one document"},
                "k": {"type": "integer", "default": 6},
            },
            "required": ["query"],
        },
    )]


def search_document(arguments: dict) -> list[types.TextContent]:
    """The tool body (sync, directly unit-testable)."""
    try:
        retriever = HybridRetriever(INDEX, doc_id=arguments.get("doc_id"))
        hits = retriever.retrieve(arguments["query"], k=int(arguments.get("k", 6)))
        return [types.TextContent(type="text", text=json.dumps(hits, ensure_ascii=False))]
    except Exception as e:                              # tool errors go back as text
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "search_document":
        raise ValueError(f"Tool not found: {name}")
    return search_document(arguments)


session_manager = StreamableHTTPSessionManager(app=mcp_server, event_store=None,
                                               stateless=True)


async def handle_streamable_http(scope, receive, send) -> None:
    await session_manager.handle_request(scope, receive, send)


@contextlib.asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    async with session_manager.run():
        logger.info("finsight MCP retrieval server up (StreamableHTTP, stateless)")
        yield


app = Starlette(routes=[Mount("/mcp", app=handle_streamable_http)], lifespan=lifespan)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.mcp_port)
