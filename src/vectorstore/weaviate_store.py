from __future__ import annotations

import os
from typing import Any

import litellm
import weaviate
from weaviate.classes.config import DataType, Property
from langchain_core.documents import Document


DEFAULT_COLLECTION_NAME = "AdobeChunks"

EMBEDDING_BATCH_SIZE = 100


class WeaviateStore:
    """Manages a Weaviate collection for storing and querying enriched chunks."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        weaviate_url: str = "http://localhost:8080",
        embedding_model: str | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._client = weaviate.connect_to_local(
            host=weaviate_url.replace("http://", "").split(":")[0],
            port=int(weaviate_url.rsplit(":", 1)[-1]),
        )
        deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
        )
        self._embedding_model = embedding_model or f"azure/{deployment}"
        self._api_base = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        if self._client.collections.exists(self._collection_name):
            return

        self._client.collections.create(
            name=self._collection_name,
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="context", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
            ],
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using litellm in batches."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            response = litellm.embedding(
                model=self._embedding_model,
                api_base=self._api_base,
                input=batch,
            )
            all_embeddings.extend([d["embedding"] for d in response.data])
        return all_embeddings

    def ingest(self, chunks: list[Document]) -> int:
        """Embed and insert chunks into Weaviate. Skips if collection already has data."""
        collection = self._client.collections.get(self._collection_name)
        existing = collection.aggregate.over_all(total_count=True).total_count
        if existing > 0:
            print(f"Collection already has {existing} objects — skipping ingestion.")
            return 0

        texts = [c.page_content for c in chunks]
        print(f"Embedding {len(texts)} chunks...")
        vectors = self._embed_texts(texts)

        collection = self._client.collections.get(self._collection_name)

        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                batch.add_object(
                    properties={
                        "content": chunk.page_content,
                        "context": chunk.metadata.get("context", ""),
                        "source": chunk.metadata.get("source", ""),
                    },
                    vector=vector,
                )

        count = collection.aggregate.over_all(total_count=True).total_count
        print(f"Ingested {len(chunks)} chunks. Collection total: {count}")
        return len(chunks)

    def search(
        self,
        query: str,
        limit: int = 5,
        alpha: float = 0.75,
    ) -> list[dict[str, Any]]:
        """Hybrid search: vector (alpha) + keyword (1-alpha)."""
        query_vector = self._embed_texts([query])[0]
        collection = self._client.collections.get(self._collection_name)

        results = collection.query.hybrid(
            query=query,
            vector=query_vector,
            alpha=alpha,
            limit=limit,
        )

        return [
            {
                "content": obj.properties.get("content", ""),
                "context": obj.properties.get("context", ""),
                "source": obj.properties.get("source", ""),
            }
            for obj in results.objects
        ]

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        if self._client.collections.exists(self._collection_name):
            self._client.collections.delete(self._collection_name)
            print(f"Deleted collection '{self._collection_name}'.")

    def close(self) -> None:
        """Close the Weaviate client connection."""
        self._client.close()
