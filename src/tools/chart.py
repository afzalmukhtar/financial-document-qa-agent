from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


CHARTS_DIR = Path("data/charts")

CHART_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_chart",
        "description": (
            "Create a chart (bar, line, or pie) from structured data and save it as a PNG. "
            "Use this when the user asks for a visual representation of financial data, trends, or comparisons."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "pie"],
                    "description": "Type of chart to create.",
                },
                "title": {
                    "type": "string",
                    "description": "Chart title.",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "X-axis labels or pie slice labels (e.g. fiscal years, segments).",
                },
                "datasets": {
                    "type": "array",
                    "description": "One or more data series. Each has a name and values matching the labels.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Series name (e.g. 'Revenue', 'Operating Expenses').",
                            },
                            "values": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Numeric values corresponding to each label.",
                            },
                        },
                        "required": ["name", "values"],
                    },
                },
                "y_label": {
                    "type": "string",
                    "description": "Y-axis label (e.g. '$ Billions'). Optional for pie charts.",
                },
            },
            "required": ["chart_type", "title", "labels", "datasets"],
        },
    },
}


def chart_tool(
    chart_type: str,
    title: str,
    labels: list[str],
    datasets: list[dict],
    y_label: str = "",
) -> str:
    """Create a chart and save to disk. Returns the file path."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "bar":
        x = range(len(labels))
        width = 0.8 / max(len(datasets), 1)
        for i, ds in enumerate(datasets):
            offset = (i - len(datasets) / 2 + 0.5) * width
            bars = ax.bar(
                [xi + offset for xi in x], ds["values"], width, label=ds["name"]
            )
            for bar, val in zip(bars, ds["values"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:,.2f}" if isinstance(val, float) else str(val),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.legend()

    elif chart_type == "line":
        for ds in datasets:
            ax.plot(labels, ds["values"], marker="o", label=ds["name"])
            for xi, val in zip(labels, ds["values"]):
                ax.annotate(
                    f"{val:,.2f}" if isinstance(val, float) else str(val),
                    (xi, val),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                )
        ax.legend()

    elif chart_type == "pie":
        values = datasets[0]["values"]
        ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)

    ax.set_title(title, fontsize=14, fontweight="bold")
    if y_label and chart_type != "pie":
        ax.set_ylabel(y_label)

    plt.tight_layout()

    safe_title = "".join(c if c.isalnum() or c in "-_ " else "" for c in title)
    safe_title = safe_title.strip().replace(" ", "_")[:60]
    filepath = CHARTS_DIR / f"{safe_title}.png"
    fig.savefig(filepath, dpi=150)
    plt.show()

    return f"Chart saved to {filepath}"
