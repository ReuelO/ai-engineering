import os
from pathlib import Path

from dotenv import load_dotenv
from llm import LLMClient

# --------------------------------------------------
# Configuration
# --------------------------------------------------
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


# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = LLMClient(
    api_key=api_key,
    model=model,
)


# --------------------------------------------------
# Conversation
# --------------------------------------------------
messages = [
    {
        "role": "system",
        "content": system_prompt,
    }
]

print("LLM CLI")
print("Commands: /reset, /exit\n")


try:
    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("Goodbye!")
            break

        if user_input.lower() == "/reset":
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ]

            print("Conversation reset.\n")
            continue

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        answer = llm.chat(messages)

        if answer is None:
            messages.pop()
            continue

        messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        print(f"AI: {answer}\n")

finally:
    llm.close()
