import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Configuration
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROJECT_DIR.parents[1]

sys.path.insert(0, str(PROJECT_DIR))

load_dotenv(PROJECT_ROOT / ".env")


from documents import load_documents
from embeddings import EmbeddingModel
from evaluation.judge import RAGJudge
from llm import RAGLLM
from rag import RAGPipeline
from search import SemanticSearch

INDEX_DIR = PROJECT_DIR / "data" / "index"
QUESTIONS_FILE = PROJECT_DIR / "evaluation" / "questions.json"
RESULTS_DIR = PROJECT_DIR / "evaluation" / "results"
RESULTS_FILE = RESULTS_DIR / "latest.json"


# Data Loading
def load_questions() -> list[dict]:
    """Load evaluation questions from JSON."""

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# Search
def build_search() -> SemanticSearch:
    """Load the existing semantic search index."""

    embedder = EmbeddingModel()

    search = SemanticSearch(embedder)

    embeddings_file = INDEX_DIR / "embeddings.npy"
    metadata_file = INDEX_DIR / "metadata.npy"

    if embeddings_file.exists() and metadata_file.exists():
        search.load(INDEX_DIR)

    else:
        documents = load_documents()

        search.index(documents)
        search.save(INDEX_DIR)

    return search


# RAG
def build_rag() -> RAGPipeline:
    """Build the RAG pipeline."""

    search = build_search()

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


# Context
def build_context(results) -> str:
    """Build LLM context from retrieved search results."""

    parts = []

    for result in results:
        parts.append(f"Source: {result.source}\nContent: {result.content}")

    return "\n\n".join(parts)


# Main Evaluation
def main():
    questions = load_questions()

    rag = build_rag()

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv(
        "OPENROUTER_MODEL",
        "openrouter/free",
    )

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    judge = RAGJudge(
        api_key=api_key,
        model=model,
    )

    # Evaluation metrics
    total = 0
    grounded = 0
    relevant = 0
    correct = 0
    total_score = 0

    # Detailed results
    results = []

    print("RAG Evaluation")
    print("=" * 60)

    for number, item in enumerate(
        questions,
        start=1,
    ):
        question = item["question"]

        print(f"\n[{number}] {question}")

        # Generate RAG answer
        response = rag.answer(
            question=question,
            top_k=3,
        )

        if response is None:
            print("RAG generation failed.")
            continue

        # Build context for judge
        context = build_context(response.sources)

        # Evaluate answer
        evaluation = judge.evaluate(
            question=question,
            context=context,
            answer=response.answer,
        )

        if evaluation is None:
            print("Judge evaluation failed.")
            continue

        # Update metrics
        total += 1

        if evaluation.grounded:
            grounded += 1

        if evaluation.relevant:
            relevant += 1

        if evaluation.correct:
            correct += 1

        total_score += evaluation.score

        # Display result
        print(f"\nAnswer:\n{response.answer}")

        print("\nSources:")

        for source in response.sources:
            print(f"- {source.source}")

        print("\nEvaluation:")
        print(f"Grounded: {evaluation.grounded}")
        print(f"Relevant: {evaluation.relevant}")
        print(f"Correct: {evaluation.correct}")
        print(f"Score: {evaluation.score}/5")
        print(f"Reasoning: {evaluation.reasoning}")

        # Store detailed result
        results.append(
            {
                "question": question,
                "answer": response.answer,
                "sources": [source.source for source in response.sources],
                "grounded": evaluation.grounded,
                "relevant": evaluation.relevant,
                "correct": evaluation.correct,
                "score": evaluation.score,
                "reasoning": evaluation.reasoning,
            }
        )

    # Evaluation Summary
    print("\n")
    print("=" * 60)
    print("RAG Evaluation Summary")
    print("=" * 60)

    if total == 0:
        print("No evaluations completed.")
        return

    print(f"Questions evaluated: {total}")

    print(f"Grounded: {grounded}/{total} ({grounded / total:.1%})")

    print(f"Relevant: {relevant}/{total} ({relevant / total:.1%})")

    print(f"Correct: {correct}/{total} ({correct / total:.1%})")

    print(f"Average score: {total_score / total:.2f}/5")

    # Save Results
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nDetailed results saved to:\n{RESULTS_FILE}")


if __name__ == "__main__":
    main()
