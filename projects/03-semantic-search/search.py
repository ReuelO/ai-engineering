from embeddings import EmbeddingModel


class SemanticSearch:
    """Search documents using embedding similarity."""

    def __init__(self, embedder: EmbeddingModel):
        self.embedder = embedder
        self.documents: list[str] = []
        self.embeddings = None

    def index(self, documents: list[str]):
        """Create embeddings for the document collection."""

        self.documents = documents
        self.embeddings = self.embedder.embed(documents)

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Return the most similar documents."""

        query_embedding = self.embedder.embed([query])

        similarities = self.embedder.model.similarity(
            query_embedding,
            self.embeddings,
        )[0]

        ranked_indices = similarities.argsort(descending=True)

        results = []

        for index in ranked_indices[:top_k]:
            index = int(index)
            score = float(similarities[index])

            results.append((self.documents[index], score))

        return results
