from __future__ import annotations


CALCULATOR_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression and return the exact numeric result. "
            "Use this for any arithmetic: percentages, growth rates, differences, ratios, etc. "
            "The expression must be valid Python math (e.g. '21505 / 19409 - 1' for growth rate)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python math expression to evaluate, e.g. '(21505 - 19409) / 19409 * 100'.",
                },
            },
            "required": ["expression"],
        },
    },
}

ALLOWED_NAMES = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
}


def calculator_tool(expression: str) -> str:
    """Safely evaluate a math expression and return the result as a string."""
    try:
        result = eval(expression, {"__builtins__": {}}, ALLOWED_NAMES)
        if isinstance(result, float):
            return str(round(result, 6))
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"
