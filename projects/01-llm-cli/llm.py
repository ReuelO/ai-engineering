from openrouter import OpenRouter


class LLMClient:
    """Small wrapper around the OpenRouter client."""

    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = OpenRouter(api_key=api_key)

    def chat(self, messages: list[dict]) -> str | None:
        """Send messages to the LLM and return its response."""

        try:
            response = self.client.chat.send(
                model=self.model,
                messages=messages,
            )

            return response.choices[0].message.content

        except Exception as error:  # noqa: BLE001
            print(f"\nLLM request failed: {error}\n")
            return None
