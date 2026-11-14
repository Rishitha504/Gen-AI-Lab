
import os
from dotenv import load_dotenv

from llm_client import (
    GeminiProvider,
    OllamaProvider,
    LLMClient
)

from mongo_logger import MongoLogger


# --------------------------------
# 1. Load Environment Variables
# --------------------------------

load_dotenv(override=True)


# --------------------------------
# 2. Display Application Menu
# --------------------------------

print("================================")
print("       WEEK 3: LLM MONITORING")
print("================================")

print("1. Gemini")
print("2. Ollama")

choice = input("\nSelect provider (1/2): ").strip()

logger = None


try:

    # --------------------------------
    # 3. Connect to MongoDB
    # --------------------------------

    logger = MongoLogger()

    # --------------------------------
    # 4. Select AI Provider
    # --------------------------------

    if choice == "1":

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "Gemini API key not found in .env file."
            )

        provider = GeminiProvider(api_key)

    elif choice == "2":

        provider = OllamaProvider("llama3.2")

    else:

        raise ValueError(
            "Invalid selection. Please choose 1 or 2."
        )

    # --------------------------------
    # 5. Initialize Common LLM Client
    # --------------------------------

    client = LLMClient(provider, logger)

    # --------------------------------
    # 6. Get User Prompt
    # --------------------------------

    prompt = input("\nEnter your question: ").strip()

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    # --------------------------------
    # 7. Generate AI Response
    # --------------------------------

    answer = client.generate(prompt)

    print("\n========== AI RESPONSE ==========")
    print(answer)

    print("\n========== COMPLETED ==========")


except Exception as error:

    print("\nApplication error:", error)


finally:

    # --------------------------------
    # 8. Close MongoDB Connection
    # --------------------------------

    if logger is not None:
        logger.close()
