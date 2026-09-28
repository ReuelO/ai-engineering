# LLM CLI

A minimal command-line AI application built with Python and OpenRouter.

The project demonstrates the fundamental architecture of an LLM-powered application, including API authentication, conversation state, system prompts, error handling, and separation of application logic from the LLM integration.

## Architecture

```text
User
 │
 ▼
Python CLI
 │
 ▼
Conversation State
 │
 ▼
LLM Client
 │
 ▼
OpenRouter API
 │
 ▼
LLM
 │
 ▼
AI Response
```

## Features

- Interactive command-line chat
- Multi-turn conversation history
- Configurable system prompt
- Configurable LLM model
- Conversation reset with `/reset`
- Exit with `/exit`
- Basic API error handling
- Environment-based configuration
- Separate LLM client abstraction

## Technologies

- Python
- OpenRouter Python SDK
- python-dotenv

## Project Structure

```text
01-llm-cli/
├── main.py
├── llm.py
├── README.md
└── requirements.txt
```

## Configuration

Create a `.env` file in the root of the `ai-engineering` repository:

```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openrouter/free
SYSTEM_PROMPT=You are a helpful AI engineering tutor. Explain technical concepts clearly and practically.
```

Never commit your API key to Git.

## Running the Project

From the repository root:

```bash
python projects/01-llm-cli/main.py
```

Available commands:

```text
/reset
/exit
```

## Concepts Demonstrated

### LLM API

Python sends messages to an LLM through an API and receives a generated response.

### Conversation State

The application maintains a list of messages and sends the relevant history with each request.

### Message Roles

The project uses:

- `system` — assistant instructions
- `user` — user input
- `assistant` — model responses

### Separation of Concerns

`main.py` manages the application while `llm.py` manages communication with the LLM provider.

### Error Handling

API failures are caught so that a failed request does not immediately terminate the application.

## Learning Outcome

This project establishes the basic pattern used by larger AI applications:

```text
Application
    ↓
Prompt + Context
    ↓
LLM
    ↓
Generated Output
    ↓
Application
```

Later projects will extend this architecture with structured output, embeddings, retrieval, tool calling, agents, evaluation, and production application patterns.
