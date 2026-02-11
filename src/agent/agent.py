from __future__ import annotations

import json
import os
from typing import Any

import litellm

from tools.calculator import CALCULATOR_TOOL_SCHEMA, calculator_tool
from tools.chart import CHART_TOOL_SCHEMA, chart_tool
from tools.retrieval import RETRIEVAL_TOOL_SCHEMA, retrieval_tool
from vectorstore import WeaviateStore

SYSTEM_PROMPT = """You are an expert financial analyst assistant specializing in Adobe Inc.'s SEC 10-K filings (fiscal years 2022-2025).

WHEN TO USE THE RETRIEVAL TOOL:
- Use retrieve_chunks whenever the user asks about Adobe's financials, operations, risks, strategy, segments, products, or any factual data from the 10-K filings.
- Do NOT use the tool for greetings, clarifications, or general knowledge questions unrelated to Adobe's filings.

QUERY DECOMPOSITION:
- If the user's question is complex or multi-part, break it into simpler sub-queries and call retrieve_chunks MULTIPLE TIMES with focused queries.
- Example: "Compare revenue and operating expenses between 2023 and 2024" should become separate retrievals like "Adobe total revenue fiscal 2023 and 2024" and "Adobe operating expenses fiscal 2023 and 2024".
- Prefer specific, targeted queries over broad ones for better retrieval accuracy.

CALCULATOR TOOL:
- Use the calculate tool for ANY arithmetic: growth rates, percentage changes, differences, ratios, totals.
- NEVER do math in your head. Always use the calculator for accuracy.
- Example: to compute YoY growth from $19.41B to $21.51B, call calculate with expression '(21510 - 19410) / 19410 * 100'.

CHART TOOL:
- Use create_chart when the user asks for a visual, trend chart, comparison chart, or any graphical representation.
- Extract the data from retrieved chunks first, then call create_chart with structured labels and datasets.
- Supported types: bar, line, pie.

ANSWERING:
- Base your answer ONLY on the retrieved chunks. Do not make up facts.
- Cite the source document (e.g. adobe_10k_fy2024.md) when referencing specific data.
- If the retrieved information does not contain the answer, say so clearly.
- Be concise, accurate, and cite specific numbers when available."""

MAX_TOOL_ROUNDS = 5


class Agent:
    """Tool-calling agent for Adobe 10-K Q&A using litellm."""

    def __init__(
        self,
        store: WeaviateStore,
        model: str | None = None,
    ) -> None:
        self._store = store
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
        self._model = model or f"azure/{deployment}"
        self._api_base = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self._tools = [RETRIEVAL_TOOL_SCHEMA, CALCULATOR_TOOL_SCHEMA, CHART_TOOL_SCHEMA]
        self._tool_map: dict[str, Any] = {
            "retrieve_chunks": self._handle_retrieval,
            "calculate": self._handle_calculate,
            "create_chart": self._handle_chart,
        }

    def _handle_retrieval(self, arguments: dict[str, Any]) -> str:
        """Dispatch a retrieve_chunks tool call."""
        return retrieval_tool(
            store=self._store,
            query=arguments["query"],
            limit=arguments.get("limit", 5),
        )

    def _handle_calculate(self, arguments: dict[str, Any]) -> str:
        """Dispatch a calculate tool call."""
        return calculator_tool(expression=arguments["expression"])

    def _handle_chart(self, arguments: dict[str, Any]) -> str:
        """Dispatch a create_chart tool call."""
        return chart_tool(
            chart_type=arguments["chart_type"],
            title=arguments["title"],
            labels=arguments["labels"],
            datasets=arguments["datasets"],
            y_label=arguments.get("y_label", ""),
        )

    def _dispatch_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Route a tool call to the correct handler."""
        handler = self._tool_map.get(name)
        if handler is None:
            return f"Unknown tool: {name}"
        return handler(arguments)

    def run(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        """Run the agent loop for a user query. Returns the final text answer.

        Args:
            query: The user's question.
            history: Optional list of previous message dicts (role/content pairs).
                     Empty by default — the caller is responsible for building
                     and passing history.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": query})

        for round_num in range(MAX_TOOL_ROUNDS):
            response = litellm.completion(
                model=self._model,
                api_base=self._api_base,
                messages=messages,
                tools=self._tools,
                temperature=0.0,
            )

            choice = response.choices[0]

            # If no tool calls, return the final answer
            if not choice.message.tool_calls:
                return choice.message.content or ""

            # Process tool calls
            messages.append(choice.message.model_dump())

            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                print(f"  [Tool] {fn_name}({json.dumps(fn_args, indent=None)})")
                result = self._dispatch_tool(fn_name, fn_args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        # Fallback if max rounds exceeded
        return "I was unable to produce a final answer within the allowed tool rounds."
