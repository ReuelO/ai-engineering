from documents import load_documents
from embeddings import EmbeddingModel
from search import SemanticSearch

embedder = EmbeddingModel()

search = SemanticSearch(embedder)

documents = load_documents()

search.index(documents)

print("Semantic Search")
print("Type /exit to quit.\n")

while True:
    query = input("Search: ").strip()

    if not query:
        continue

    if query.lower() == "/exit":
        print("Goodbye!")
        break

    results = search.search(
        query=query,
        top_k=3,
    )

    print("\nResults:")

    for document, score in results:
        print(f"\nScore: {score:.4f}")
        print(document[:300])

    print()
