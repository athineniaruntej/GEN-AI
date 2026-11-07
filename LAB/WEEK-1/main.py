
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors

# Step 1: Load environment variables
load_dotenv()

# Step 2: Read Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Step 3: Check API key
if not api_key:
    raise ValueError("Gemini API key not found!")

# Step 4: Create Gemini client
client = genai.Client(api_key=api_key)

# Step 5: Send prompt to Gemini
prompt = "What is Artificial Intelligence? Explain in simple words."

for attempt in range(3):
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        # Step 6: Display response
        print("\nLLM Response:")
        print(response.text)
        break

    except errors.ServerError as e:
        print("Gemini server is temporarily unavailable.")

        if attempt < 2:
            print("Retrying...")
            time.sleep(5 * (attempt + 1))
        else:
            print("Please try again later.")
            print(e)

    except errors.APIError as e:
        print("API Error:", e)
        break
