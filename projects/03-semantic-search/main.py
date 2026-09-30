from documents import load_documents
from embeddings import EmbeddingModel
from search import SemanticSearch

embedder = EmbeddingModel()

search = SemanticSearch(embedder)

documents = load_documents()

search.index(documents)

print("Semantic Search")
print(f"Indexed {len(documents)} document chunks.")
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

    for result in results:
        print(
            f"\n[{result.source} | chunk {result.chunk_id} | score {result.score:.4f}]"
        )

        print(result.content)

    print()
