# LLM CLI

A bare-bones CLI chat app using Python and OpenRouter. 

This is project #1 in the `ai-engineering` series. It's meant to show the absolute basics of how an LLM app actually works under the hood—handling API keys, keeping track of chat history, and keeping the LLM logic separate from the UI.

## Architecture

```text
User → Python CLI → Conversation State → LLM Client → OpenRouter API → LLM
```
*(It's just a loop: get input, format the context, call the API, print the response, repeat.)*

## Features

- Interactive terminal chat
- Multi-turn conversation history
- Customizable system prompts and models
- `/reset` to clear context, `/exit` to quit
- Basic error handling so the app doesn't just crash if the API times out
- Clean separation between the CLI logic and the LLM provider

## Tech Stack

- Python
- OpenRouter Python SDK
- python-dotenv

## Project Structure

```text
01-llm-cli/
├── main.py          # CLI loop and user input
├── llm.py           # API calls and LLM logic
├── README.md
└── requirements.txt
```

## Configuration

You'll need a `.env` file in the root of the `ai-engineering` repo. Grab an API key from OpenRouter and add it here:

```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openrouter/free
SYSTEM_PROMPT=You are a helpful AI engineering tutor. Explain technical concepts clearly and practically.
```

*(Obviously, don't commit this file to git.)*

## Running the Project

Run it from the root of the repo:

```bash
python projects/01-llm-cli/main.py
```

Once it's running, just start typing. Use `/reset` to clear the chat history or `/exit` to quit.

## How it Works

At its core, almost every LLM app is just a loop that sends a prompt + context to a model, then does something with the output. This project breaks that down into a few key concepts:

- **State management:** We keep a list of messages (`system`, `user`, `assistant`) and pass the relevant history with every request so the model remembers the conversation.
- **Separation of concerns:** `main.py` handles the terminal loop, while `llm.py` handles the actual API calls. This makes it easy to swap out the provider or change the UI later without breaking everything.
- **Graceful failures:** If the API throws an error, the app catches it and lets you try again instead of terminating.

## What's Next?

This project establishes the baseline pattern we'll use for everything else. In later projects, we'll extend this exact same architecture to add structured output, embeddings, RAG, tool calling, and agents.