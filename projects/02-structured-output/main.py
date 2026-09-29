import json
import os
from pathlib import Path

from dotenv import load_dotenv
from llm import StructuredLLM
from schemas import TopicAnalysis

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("OPENROUTER_MODEL", "openrouter/free")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set.")


# LLM
llm = StructuredLLM(
    api_key=api_key,
    model=model,
)


# Input
print("Topic Analyzer")
print("Type /exit to quit.\n")

while True:
    topic = input("Enter a topic: ").strip()

    if not topic:
        continue

    if topic.lower() == "/exit":
        print("Goodbye!")
        break

    # Generate structured response
    raw_output = llm.analyze_topic(
        topic=topic,
        schema=TopicAnalysis.model_json_schema(),
    )

    if raw_output is None:
        continue

    # Parse JSON
    try:
        data = json.loads(raw_output)

    except json.JSONDecodeError as error:
        print("\nThe model returned invalid JSON.")
        print(f"Error: {error}")
        print(f"\nRaw response:\n{raw_output}\n")
        continue

    # Validate response
    try:
        analysis = TopicAnalysis.model_validate(data)

    except Exception as error:  # noqa: BLE001
        print("\nThe model returned invalid structured data.")
        print(f"Error: {error}")
        print(f"\nRaw response:\n{raw_output}\n")
        continue

    # Display
    print("\nValidated result:")
    print(f"\nTopic: {analysis.topic}")
    print(f"Difficulty: {analysis.difficulty}")
    print(f"\nSummary:\n{analysis.summary}")

    print("\nKey concepts:")

    for concept in analysis.key_concepts:
        print(f"- {concept}")

    print()
