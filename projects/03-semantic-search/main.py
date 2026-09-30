from embeddings import EmbeddingModel
from search import SemanticSearch

embedder = EmbeddingModel()

search = SemanticSearch(embedder)

documents = [
    "Python functions allow code to be reused.",
    "Python lists store multiple values in a single collection.",
    "SQL joins combine data from multiple database tables.",
    "Linux provides a command-line environment for managing computers.",
    "Machine learning allows computers to learn patterns from data.",
]

search.index(documents)

query = input("Search: ").strip()

results = search.search(
    query=query,
    top_k=3,
)

print("\nResults:")

for document, score in results:
    print(f"\nScore: {score:.4f}")
    print(document)
