from abc import ABC, abstractmethod

from groq import Groq
import requests


# 1. Common interface
class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass


# 2. Groq cloud provider
class GroqProvider(LLMProvider):

    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def generate(self, prompt):

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )

        return response.choices[0].message.content


# 3. Ollama local provider
class OllamaProvider(LLMProvider):

    def __init__(self, model="llama3.2"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        return response.json()["response"]


# 4. Common client
class LLMClient:

    def __init__(self, provider):
        self.provider = provider

    def generate(self, prompt):
        return self.provider.generate(prompt)