# Structured LLM Output

A small project that converts an LLM's natural-language response into validated, machine-readable data.

This is project #2 in the `ai-engineering` series. It's meant to show how to force an LLM to output strict JSON and validate it against a schema before your app ever touches it.

## Architecture

```text
User → Python App → Structured Request → OpenRouter → LLM → JSON Response → Pydantic Validation → App Logic
```

_(It's a pipeline: ask for JSON, parse it, validate it against a strict schema, and only then trust it.)_

## Features

- Forces the LLM to output strict JSON using a formal schema
- Parses and validates the raw response with Pydantic
- Creates a hard boundary between unpredictable LLM output and trusted app data
- Generates JSON schemas directly from Python type hints
- Clean separation between the LLM provider, the data models, and the CLI

## Tech Stack

- Python
- Pydantic (for data validation and schema generation)
- OpenRouter Python SDK
- python-dotenv

## Project Structure

```text
02-structured-output/
├── main.py          # CLI loop, user input, and printing results
├── llm.py           # API calls and LLM logic
├── schemas.py       # Pydantic models defining the expected JSON shape
├── README.md
└── requirements.txt
```

## Configuration

You'll need a `.env` file in the root of the `ai-engineering` repo. Grab an API key from OpenRouter and add it here:

```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openrouter/free
```

_(Obviously, don't commit this file to git.)_

## Running the Project

Run it from the root of the repo:

```bash
python projects/02-structured-output/main.py
```

Once it's running, enter a topic when prompted. The app will print out the validated data in a clean, readable format.

## How it Works

Getting reliable data out of an LLM comes down to a few key concepts:

- **Strict contracts:** Instead of just prompting "please return JSON", we pass a formal JSON schema in the API request. This gives the model a strict contract to follow.
- **Trust, but verify:** LLMs are probabilistic and can still mess up the format. We parse the raw text and validate it with Pydantic so the app only ever works with trusted, typed Python objects.
- **Separation of concerns:** Just like the CLI, `llm.py` handles the API, but now `schemas.py` handles the data contract. The app logic doesn't need to know how the LLM works, just that it returns a valid `TopicAnalysis` object.

## What's Next?

This project establishes how to get reliable, structured data out of an LLM. In the next project, we'll take that concept and apply it to embeddings and semantic search.
