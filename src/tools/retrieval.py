from __future__ import annotations

from vectorstore import WeaviateStore


RETRIEVAL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_chunks",
        "description": (
            "Search the Adobe 10-K vector store for relevant document chunks. "
            "Use this tool whenever the user asks a question about Adobe's financials, "
            "business segments, risk factors, revenue, or any content from SEC 10-K filings."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language search query to find relevant chunks.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of chunks to return. Defaults to 5.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


def retrieval_tool(store: WeaviateStore, query: str, limit: int = 5) -> str:
    """Search the vector store and return formatted results for the LLM."""
    results = store.search(query=query, limit=limit)

    if not results:
        return "No relevant chunks found for the given query."

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Result {i}] (source: {r['source']})\n{r['content']}"
        )

    return "\n\n---\n\n".join(parts)
