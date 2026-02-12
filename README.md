# Adobe 10-K Financial Document Q&A

An AI-powered agent that ingests Adobe's SEC 10-K filings and answers leadership questions about the company's performance, risks, and strategy, grounded entirely in the source documents.

---

## Setup

### 1. Prerequisites

| Requirement | Version | Install |
|---|---|---|
| **Python** | 3.11+ | [python.org](https://www.python.org/downloads/) |
| **uv** | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Docker** | latest | See below |

**Docker installation by OS:**

- **macOS:** [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/) (or [Colima](https://github.com/abiosoft/colima) as a lightweight alternative)
- **Windows:** [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- **Linux:** [Docker Engine](https://docs.docker.com/engine/install/)

### 2. Start Weaviate (Vector Database)

Weaviate runs locally via Docker. No accounts or API keys needed.

```bash
docker compose up -d
```

Verify it is running:

```bash
curl http://localhost:8080/v1/.well-known/ready
```

Data is persisted in a Docker volume (`weaviate_data`) and survives container restarts.

```bash
docker compose down          # stop, keep data
docker compose down -v       # stop and delete all data
```

### 3. Install Python Dependencies

```bash
uv sync
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your Azure OpenAI credentials:

| Variable | Description | Example |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource URL | `https://your-resource.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | API key for the resource | `your-api-key` |
| `AZURE_OPENAI_API_VERSION` | API version | `2024-08-01-preview` |
| `AZURE_OPENAI_DEPLOYMENT` | Chat model deployment name | `gpt-4.1` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding model deployment name | `text-embedding-3-large` |

### 5. Run

Open `demo.ipynb` in Jupyter or VS Code and **Run All Cells**. The notebook will:

1. Download and clean Adobe 10-K HTML filings from SEC EDGAR
2. Chunk documents with LLM-generated contextual enrichment
3. Embed and ingest chunks into Weaviate
4. Start the agent and run example queries (including charts)

> **Note:** The raw HTML filings, cleaned markdown files, and the LLM-generated chunk context cache are all committed to this repo. On a fresh run the notebook automatically detects existing data and skips the download, cleaning, and context-generation steps, so you only need a running Weaviate instance and valid Azure OpenAI credentials to get started.

---

## Project Structure

```
├── demo.ipynb                  # Main notebook, run this
├── docker-compose.yml          # Weaviate local setup
├── pyproject.toml              # Dependencies & build config
├── .env.example                # Environment variable template
│
├── src/
│   ├── utils/
│   │   └── helpers.py          # HTML cleaning & markdown conversion
│   ├── chunker/
│   │   └── contextual_chunker.py  # Token-based chunking + LLM context
│   ├── vectorstore/
│   │   └── weaviate_store.py   # Weaviate embed, ingest, hybrid search
│   ├── tools/
│   │   ├── retrieval.py        # Retrieval tool (searches vector store)
│   │   ├── calculator.py       # Math calculator tool (safe eval)
│   │   └── chart.py            # Chart generation tool (bar, line, pie)
│   └── agent/
│       └── agent.py            # Tool-calling agent loop (litellm)
│
├── data/
│   ├── html/                   # Raw SEC EDGAR HTML filings
│   ├── *.md                    # Cleaned markdown files
│   ├── .chunk_cache/           # SHA-256 context cache
│   └── charts/                 # Generated chart PNGs
```

---

## Assumptions

- **Data Source:** No dataset was provided by Adobe. I used publicly available Adobe 10-K annual filings (fiscal years 2022–2025) from SEC EDGAR.
- **Scope:** The agent handles textual data and financial tables from the filings. It also generates charts/plots (bar, line, pie) when requested.
- **Output Format:** Natural-language answers grounded in the source documents, with optional chart generation for visual queries.
- **Model Interface:** [LiteLLM](https://github.com/BerriAI/litellm) abstracts the model layer, making the system LLM-agnostic. Swap the model by changing the `AZURE_OPENAI_DEPLOYMENT` env var; no code changes needed.
- **LLM used:** Azure OpenAI `gpt-4.1` for chat/tool-calling and contextual chunk generation.
- **Embedding model used:** Azure OpenAI `text-embedding-3-large` (3072 dimensions).
- **Tokenizer:** `cl100k_base` tiktoken encoding (matches GPT-4 / GPT-4.1 tokenizer).
- **Orchestration:** LangChain is used for document loading and recursive text splitting; the agent loop itself is a lightweight custom implementation using LiteLLM's tool-calling API.
- **Retrieval:** Weaviate vector store with built-in hybrid search (vector + BM25 keyword) to handle precise financial terminology that pure semantic search can miss.

## Design Decisions

### 1. PDF to HTML for Document Ingestion
- **Problem:** PDF extraction via PyMuPDF/pymupdf4llm produced garbled text on financial tables due to custom font encodings in SEC filings.
- **Solution:** Download original HTML filings from SEC EDGAR using `edgartools`. HTML is the native format, no lossy PDF-to-text conversion.
- **Benefit:** Faster processing, no OCR needed, exact text as filed.

### 2. HTML Cleaning Before Markdown Conversion
- **Why:** Raw SEC HTML contains scripts, styles, XBRL inline tags (`ix:*`), hidden elements, and layout attributes that add noise.
- **What we remove:** `script`, `style`, `noscript`, `meta`, `link`, `header`, `footer`, `nav`, `aside`, `iframe`, `object`, `embed`, `ix:header`, `display:none` elements.
- **What we keep:** XBRL inline tag *content* (unwrapped), it contains the actual financial data.
- **Result:** Clean markdown with preserved table structure.
- **Implementation:** `clean_and_convert_html_2_markdown()` in `src/utils/helpers.py` (BeautifulSoup + html2text).

### 3. Recursive Token-Based Chunking
- **Why not character-based:** Token counts matter for LLM context windows and embedding models. Character splitting produces inconsistent token counts.
- **Approach:** LangChain `RecursiveCharacterTextSplitter` with `tiktoken` length function. Splits on `["\n\n", "\n", ".", "?", "!", " "]`, prioritizing paragraph and sentence boundaries.
- **Chunk size:** 1024 tokens, 128-token overlap, tuned for context retention vs. retrieval granularity.
- **No special table handling:** Recursive chunking handles markdown tables reasonably, and contextual enrichment compensates for lost structure.

### 4. LLM-Generated Contextual Enrichment
- **Problem:** Chunks lose context, "revenue increased 12%" does not say which company, year, or segment. Hurts embedding retrieval accuracy.
- **Solution:** For each chunk, a sliding window (250 tokens before + after) is sent to an LLM to generate 3-5 sentences of context prepended as `[Chunk_Context_Start]...[Chunk_Context_End]`.
- **Context structure:** 1-2 sentences on what the preceding text discussed and how this chunk relates, 1 sentence summarizing the chunk itself, 1-2 sentences on what comes next.
- **Structured output via tool calling:** Pydantic model `ChunkContext` with a single `context` field. Tool calling forces structured JSON, consistent, parseable output.
- **LLM:** Azure OpenAI via `litellm`.

**Example** -- a raw chunk after splitting:

> *"We employ our product-led growth strategy to minimize the friction of customer interactions and drive positive product experiences, which results in increasing adoption, usage, conversion, expansion and loyalty..."*

This chunk never mentions "Adobe", "Creative Cloud", or the fiscal year. The generated context prepended to it:

> `[Chunk_Context_Start]` *The preceding text discussed Adobe's ongoing improvements and new features in its Digital Media products. This chunk continues by detailing Adobe's product-led growth and data-driven strategies for its Creative Cloud and Document Cloud businesses, focusing on customer acquisition, retention, and revenue growth for fiscal year 2023.* `[Chunk_Context_End]`

The enrichment injects "Adobe", "Creative Cloud", "Document Cloud", and "fiscal year 2023" so the embedding captures the full meaning and retrieval accuracy improves.

### 5. Weaviate Vector Store
- **Why Weaviate:** Open-source, supports hybrid search (vector + keyword), runs locally via Docker with zero config.
- **Setup:** Single `docker compose up -d`, no accounts, no API keys needed for local dev.
- **Persistence:** Data stored in a Docker volume (`weaviate_data`), survives container restarts.

### 6. Hash-Based Context Cache
- **Problem:** ~400+ chunks x 1 LLM call each is slow. Re-running should not repeat work.
- **Solution:** SHA-256 hash of each chunk mapped to its context in `data/.chunk_cache/cache.json`. Subsequent runs load from cache instantly; only new/changed chunks trigger LLM calls.
- **Motivation:** Primarily added so context generation doesn't slow down testing and iteration. To regenerate all contexts from scratch, change the chunker configuration (e.g. chunk size or overlap) and the cache misses will trigger fresh LLM calls.

### 7. Future Speed Improvements
- Context generation is synchronous. For production, `litellm.acompletion` with `asyncio.gather` and a concurrency semaphore (~10 parallel) would reduce context generation time ~10x.
