from dataclasses import dataclass

from documents import DocumentChunk
from embeddings import EmbeddingModel


@dataclass
class SearchResult:
    """A ranked semantic search result."""

    content: str
    source: str
    chunk_id: int
    score: float


class SemanticSearch:
    """Search document chunks using embedding similarity."""

    def __init__(self, embedder: EmbeddingModel):
        self.embedder = embedder
        self.documents: list[DocumentChunk] = []
        self.embeddings = None

    def index(self, documents: list[DocumentChunk]):
        """Create embeddings for document chunks."""

        self.documents = documents

        texts = [document.content for document in documents]

        self.embeddings = self.embedder.embed(texts)

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Return the most similar document chunks."""

        query_embedding = self.embedder.embed([query])

        similarities = self.embedder.model.similarity(
            query_embedding,
            self.embeddings,
        )[0]

        ranked_indices = similarities.argsort(descending=True)

        results = []

        for index in ranked_indices[:top_k]:
            index = int(index)

            document = self.documents[index]

            results.append(
                SearchResult(
                    content=document.content,
                    source=document.source,
                    chunk_id=document.chunk_id,
                    score=float(similarities[index]),
                )
            )

        return results
