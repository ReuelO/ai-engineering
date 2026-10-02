import os
from pathlib import Path

from documents import load_documents
from dotenv import load_dotenv
from embeddings import EmbeddingModel
from llm import RAGLLM
from rag import RAGPipeline
from search import SemanticSearch

PROJECT_DIR = Path(__file__).parent
PROJECT_ROOT = PROJECT_DIR.parents[1]

INDEX_DIR = PROJECT_DIR / "data" / "index"

load_dotenv(PROJECT_ROOT / ".env")


api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free",
)

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set.")


# Embedding model
embedder = EmbeddingModel()

# Semantic search
search = SemanticSearch(embedder)


# Build or load search index
if (INDEX_DIR / "embeddings.npy").exists() and (INDEX_DIR / "metadata.npy").exists():
    print("Loading existing search index...")
    search.load(INDEX_DIR)

else:
    print("Building search index...")

    documents = load_documents()

    search.index(documents)
    search.save(INDEX_DIR)

    print("Search index created.")


# LLM
llm = RAGLLM(
    api_key=api_key,
    model=model,
)


# RAG pipeline
rag = RAGPipeline(
    search=search,
    llm=llm,
)


print()
print("RAG Knowledge Assistant")
print(f"Indexed {len(search.documents)} chunks.")
print("Type /exit to quit.\n")


while True:
    question = input("You: ").strip()

    if not question:
        continue

    if question.lower() == "/exit":
        print("Goodbye!")
        break

    response = rag.answer(
        question=question,
        top_k=3,
    )

    if response is None:
        continue

    print("\nAssistant:")
    print(response.answer)

    print("\nSources:")

    for source in response.sources:
        print(f"- {source.source}, chunk {source.chunk_id}, score {source.score:.4f}")

    print()
