from dataclasses import dataclass

from llm import RAGLLM
from search import SearchResult, SemanticSearch


@dataclass
class RAGResponse:
    """A generated answer and its supporting sources."""

    answer: str
    sources: list[SearchResult]


class RAGPipeline:
    """Retrieve relevant knowledge and generate an answer."""

    def __init__(
        self,
        search: SemanticSearch,
        llm: RAGLLM,
    ):
        self.search = search
        self.llm = llm

    def answer(
        self,
        question: str,
        top_k: int = 3,
    ) -> RAGResponse | None:
        """Retrieve context and generate an answer."""

        results = self.search.search(
            query=question,
            top_k=top_k,
        )

        context_parts = []

        for result in results:
            context_parts.append(f"Source: {result.source}\nContent: {result.content}")

        context = "\n\n".join(context_parts)

        answer = self.llm.generate(
            question=question,
            context=context,
        )

        if answer is None:
            return None

        return RAGResponse(
            answer=answer,
            sources=results,
        )
