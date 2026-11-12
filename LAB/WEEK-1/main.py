import os
import time
from dotenv import load_dotenv
from groq import Groq

# Step 1: Load environment variables
load_dotenv()

# Step 2: Read Groq API key
api_key = os.getenv("GROQ_API_KEY")

# Step 3: Check API key
if not api_key:
    raise ValueError("Groq API key not found!")

# Step 4: Create Groq client
client = Groq(api_key=api_key)

# Step 5: Send prompt to Groq
prompt = "What is Artificial Intelligence? Explain in simple words."

for attempt in range(3):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )

        # Step 6: Display response
        print("\nLLM Response:")
        print(response.choices[0].message.content)
        break

    except Exception as e:
        print("Groq server is temporarily unavailable.")

        if attempt < 2:
            print("Retrying...")
            time.sleep(5 ** (attempt + 1))
        else:
            print("Please try again later.")
            print(e)