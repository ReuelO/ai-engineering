import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))


from documents import load_documents
from embeddings import EmbeddingModel
from llm import RAGLLM
from rag import RAGPipeline
from search import SemanticSearch

INDEX_DIR = PROJECT_DIR / "data" / "index"
QUESTIONS_FILE = PROJECT_DIR / "evaluation" / "questions.json"

PROJECT_ROOT = PROJECT_DIR.parents[1]


def load_questions() -> list[dict]:
    """Load answer evaluation questions."""

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_search() -> SemanticSearch:
    """Load or build the semantic search index."""

    embedder = EmbeddingModel()

    search = SemanticSearch(embedder)

    if (INDEX_DIR / "embeddings.npy").exists() and (
        INDEX_DIR / "metadata.npy"
    ).exists():
        search.load(INDEX_DIR)

    else:
        documents = load_documents()
        search.index(documents)
        search.save(INDEX_DIR)

    return search


def build_rag() -> RAGPipeline:
    """Create the RAG pipeline."""

    search = build_search()

    import os

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv(
        "OPENROUTER_MODEL",
        "openrouter/free",
    )

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    llm = RAGLLM(
        api_key=api_key,
        model=model,
    )

    return RAGPipeline(
        search=search,
        llm=llm,
    )


def main():
    questions = load_questions()
    rag = build_rag()

    print("RAG Answer Evaluation")
    print("=" * 60)

    for number, item in enumerate(
        questions,
        start=1,
    ):
        question = item["question"]
        expected_source = item["expected_source"]

        response = rag.answer(
            question=question,
            top_k=3,
        )

        if response is None:
            print(f"\n[{number}] FAILED")
            print(f"Question: {question}")
            continue

        sources = [source.source for source in response.sources]

        source_found = expected_source in sources

        print(f"\n[{number}]")
        print(f"Question: {question}")
        print(f"Expected source: {expected_source}")
        print(f"Retrieved sources: {sources}")
        print(f"Source retrieved: {source_found}")

        print("\nAnswer:")
        print(response.answer)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    main()
