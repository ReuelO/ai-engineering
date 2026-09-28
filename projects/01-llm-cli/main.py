import os
from pathlib import Path

from dotenv import load_dotenv
from llm import LLMClient

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("OPENROUTER_MODEL", "openrouter/free")

system_prompt = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful AI assistant.",
)

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set.")


# LLM Client
llm = LLMClient(
    api_key=api_key,
    model=model,
)


# Conversation
messages = [
    {
        "role": "system",
        "content": system_prompt,
    }
]

print("LLM CLI")
print("Commands: /reset, /exit\n")


while True:
    # Get user input
    user_input = input("You: ").strip()

    if not user_input:
        continue

    # Exit the program
    if user_input.lower() == "/exit":
        print("Goodbye!")
        break

    # Reset conversation
    if user_input.lower() == "/reset":
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        print("Conversation reset.\n")
        continue

    # Add user message to conversation
    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # Get LLM response
    answer = llm.chat(messages)

    if answer is None:
        messages.pop()
        continue

    # Add LLM response to conversation
    messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    # Show LLM response
    print(f"AI: {answer}\n")
