from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import litellm
import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

DEFAULT_SEPARATORS = ["\n\n", "\n", ".", "?", "!", " "]


class ChunkContext(BaseModel):
    """Structured context for a chunk of a SEC 10-K filing."""

    context: str = Field(
        description="3 to 5 sentences situating this chunk. First 1-2 sentences describe what the preceding text discussed and how this chunk relates to it. The middle sentence summarizes what this chunk itself covers. The final 1-2 sentences describe what the following text transitions into."
    )


CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "provide_chunk_context",
        "description": "Provide structured context that situates a chunk within a SEC 10-K filing.",
        "parameters": ChunkContext.model_json_schema(),
    },
}

CONTEXT_PROMPT = """
You are an expert at situating a chunk of text within a larger SEC 10-K filing.

<before_context>
{before}
</before_context>

<chunk>
{chunk}
</chunk>

<after_context>
{after}
</after_context>

Use the provide_chunk_context tool. Write 3-5 natural-language sentences (not keywords or lists) in the `context` field:
- (1-2 sentences) What the preceding text was discussing and how this chunk continues or relates to it.
- (1 sentence) A concise summary of what this chunk covers, including the company name, fiscal year, and any key terms.
- (1-2 sentences) What the text transitions into after this chunk.
"""


class ContextualChunker:
    """Recursive chunker using LangChain splitter with tiktoken token counting."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        encoding_name: str = "cl100k_base",
        context_window_tokens: int = 250,
        cache_dir: str = "data/.chunk_cache",
    ) -> None:
        self._enc = tiktoken.get_encoding(encoding_name)
        self._context_window_tokens = context_window_tokens
        self._model = f"azure/{os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4.1')}"
        self._api_base = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or DEFAULT_SEPARATORS,
            length_function=lambda text: len(self._enc.encode(text)),
            is_separator_regex=False,
        )

    def _token_len(self, text: str) -> int:
        return len(self._enc.encode(text))

    @staticmethod
    def _hash_chunk(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _load_cache(self) -> dict[str, dict]:
        cache_file = self._cache_dir / "cache.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self, cache: dict[str, dict]) -> None:
        cache_file = self._cache_dir / "cache.json"
        cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    def _apply_context(self, chunk: Document, ctx: ChunkContext) -> None:
        chunk.metadata["context"] = ctx.context
        chunk.page_content = (
            f"[Chunk_Context_Start]{ctx.context}[Chunk_Context_End]\n\n"
            f"{chunk.page_content}"
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into smaller chunks and enrich with LLM-generated context."""
        chunks = self._splitter.split_documents(documents)

        # Build source text lookup
        source_texts: dict[str, str] = {}
        for doc in documents:
            source = doc.metadata.get("source", "")
            source_texts[source] = doc.page_content

        cache = self._load_cache()
        cache_hits = 0

        for i, chunk in enumerate(chunks):
            h = self._hash_chunk(chunk.page_content)
            if h in cache:
                ctx = ChunkContext(**cache[h])
                cache_hits += 1
            else:
                source = chunk.metadata.get("source", "")
                full_text = source_texts.get(source, "")
                ctx = self._generate_context(chunk, full_text)
                cache[h] = ctx.model_dump()
            self._apply_context(chunk, ctx)
            print(
                f"  [{i + 1}/{len(chunks)}] processed (cache hits: {cache_hits})",
                end="\r",
            )

        self._save_cache(cache)
        print(
            f"\nDone: {len(chunks)} chunks ({cache_hits} from cache, "
            f"{len(chunks) - cache_hits} generated)."
        )
        return chunks

    def _get_sliding_window(self, full_text: str, chunk_text: str) -> tuple[str, str]:
        """Get ~250 tokens before and ~250 tokens after the chunk in the source doc."""
        idx = full_text.find(chunk_text)
        if idx == -1:
            return "", ""

        # Before window
        before_text = full_text[:idx]
        before_tokens = self._enc.encode(before_text)
        if len(before_tokens) > self._context_window_tokens:
            before_tokens = before_tokens[-self._context_window_tokens :]
        before = self._enc.decode(before_tokens).strip()

        # After window
        after_start = idx + len(chunk_text)
        after_text = full_text[after_start:]
        after_tokens = self._enc.encode(after_text)
        if len(after_tokens) > self._context_window_tokens:
            after_tokens = after_tokens[: self._context_window_tokens]
        after = self._enc.decode(after_tokens).strip()

        return before, after

    def _generate_context(
        self,
        chunk: Document,
        full_text: str,
    ) -> ChunkContext:
        """Generate structured context for a chunk using LLM tool calling."""
        before, after = self._get_sliding_window(full_text, chunk.page_content)

        response = litellm.completion(
            model=self._model,
            api_base=self._api_base,
            messages=[
                {
                    "role": "user",
                    "content": CONTEXT_PROMPT.format(
                        before=before,
                        chunk=chunk.page_content,
                        after=after,
                    ),
                },
            ],
            tools=[CONTEXT_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "provide_chunk_context"},
            },
            temperature=0.0,
        )
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        return ChunkContext(**args)
