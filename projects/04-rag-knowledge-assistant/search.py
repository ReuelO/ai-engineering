from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    content: str
    source: str
    score: float


class SemanticSearch:
    """Semantic search over document chunks."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.embeddings = None
        self.metadata = []

    def index(self, documents):
        """Create an embedding index."""

        texts = [document.content for document in documents]

        self.embeddings = self.embedder.embed(texts)

        self.metadata = [
            {
                "content": document.content,
                "source": document.source,
            }
            for document in documents
        ]

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Return the most similar document chunks."""

        if self.embeddings is None:
            raise RuntimeError("Search index has not been built.")

        query_embedding = self.embedder.embed([query])[0]

        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            metadata = self.metadata[index]

            results.append(
                SearchResult(
                    content=metadata["content"],
                    source=metadata["source"],
                    score=float(scores[index]),
                )
            )

        return results
