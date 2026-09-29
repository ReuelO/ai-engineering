from typing import Literal

from pydantic import BaseModel, Field


class TopicAnalysis(BaseModel):
    topic: str = Field(description="The topic being analyzed.")
    difficulty: Literal[
        "beginner",
        "intermediate",
        "advanced",
    ] = Field(description="Estimated learning difficulty.")
    summary: str = Field(description="A concise explanation of the topic.")
    key_concepts: list[str] = Field(
        description="The most important concepts associated with the topic."
    )
