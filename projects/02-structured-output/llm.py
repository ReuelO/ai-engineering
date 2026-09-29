from openrouter import OpenRouter


class StructuredLLM:
    """Client for structured LLM responses."""

    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = OpenRouter(api_key=api_key)

    def analyze_topic(self, topic: str, schema: dict) -> str | None:
        """Analyze an educational topic and return JSON."""

        try:
            response = self.client.chat.send(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You analyze educational topics. "
                            "Return ONLY a valid JSON object with "
                            "the fields topic, difficulty, summary, "
                            "and key_concepts. "
                            "difficulty must be beginner, "
                            "intermediate, or advanced. "
                            "key_concepts must contain 3 to 10 items."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this topic: {topic}",
                    },
                ],
                response_format={
                    "type": "json_object",
                },
            )

            return response.choices[0].message.content

        except Exception as error:  # noqa: BLE001
            print("\nLLM request failed.")
            print(f"Error type: {type(error).__name__}")
            print(f"Error: {error}")
            return None
