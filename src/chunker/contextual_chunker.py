from __future__ import annotations

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DEFAULT_SEPARATORS = ["\n\n", "\n", ".", "?", "!", " "]


class ContextualChunker:
    """Recursive chunker using LangChain splitter with tiktoken token counting."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self._enc = tiktoken.get_encoding(encoding_name)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or DEFAULT_SEPARATORS,
            length_function=lambda text: len(self._enc.encode(text)),
            is_separator_regex=False,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split a list of Documents into smaller chunks."""
        return self._splitter.split_documents(documents)
