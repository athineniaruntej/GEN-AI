import os

from dotenv import load_dotenv

from llm_client import GroqProvider, OllamaProvider, LLMClient
from usage_logger import UsageLogger

load_dotenv(override=True)

usage_logger = UsageLogger()

try:

    provider_name = input(
        "Choose provider (groq/ollama): "
    ).strip().lower()

    if provider_name == "groq":

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")

        provider = GroqProvider(api_key)

    elif provider_name == "ollama":

        provider = OllamaProvider("llama3.2")

    else:

        raise ValueError("Invalid provider selected")

    # Create common client
    client = LLMClient(provider, usage_logger)

    # Get user prompt
    prompt = input("Enter your question: ")

    # Generate response
    answer = client.generate(prompt)

    # Display response
    print("\nLLM Response:")
    print(answer)

finally:

    usage_logger.close()