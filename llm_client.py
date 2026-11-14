
from abc import ABC, abstractmethod
from google import genai
from google.genai import types
import requests
import time

from cost_calculator import calculate_cost


# --------------------------------
# 1. Common LLM Provider Interface
# --------------------------------

class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass


# --------------------------------
# 2. Gemini Provider
# --------------------------------

class GeminiProvider(LLMProvider):

    def __init__(self, api_key, model="gemini-3.8-flash"):

        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=120000
            )
        )

        self.model = model
        self.fallback_models = ["gemini-3.5-flash-lite", "gemini-2.5-flash"]

    def generate(self, prompt):

        models_to_try = [self.model] + [
            m for m in self.fallback_models if m != self.model
        ]

        last_error = None

        for current_model in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level="LOW"
                        )
                    )
                )

                # Update self.model so MongoDB logs the model that actually succeeded
                self.model = current_model

                usage = response.usage_metadata
                input_tokens = usage.prompt_token_count if usage else 0
                output_tokens = usage.candidates_token_count if usage else 0

                return {
                    "response": response.text or "",
                    "input_tokens": input_tokens or 0,
                    "output_tokens": output_tokens or 0
                }

            except Exception as error:
                last_error = error
                status_code = getattr(error, "status_code", getattr(error, "code", None))

                # If server is overloaded (503) or timed out (504), try the next fallback model
                if status_code in (503, 504):
                    print(f"Model {current_model} busy ({status_code}), trying fallback...")
                    continue

                raise error

        raise last_error



# --------------------------------
# 3. Ollama Provider
# --------------------------------

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

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama Error {response.status_code}: "
                f"{response.text}"
            )

        data = response.json()

        return {
            "response": data.get("response", ""),
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0)
        }


# --------------------------------
# 4. Common LLM Client
# --------------------------------

class LLMClient:

    def __init__(self, provider, logger):

        self.provider = provider
        self.logger = logger

    def generate(self, prompt):

        max_attempts = 3
        start_time = time.perf_counter()
        last_error = None

        for attempt in range(1, max_attempts + 1):

            try:

                # Generate AI response
                result = self.provider.generate(prompt)

                # Calculate latency
                latency = round(
                    (time.perf_counter() - start_time) * 1000,
                    2
                )

                input_tokens = result["input_tokens"]
                output_tokens = result["output_tokens"]

                model = self.provider.model
                provider_name = type(self.provider).__name__

                # Calculate cost
                cost = calculate_cost(
                    provider_name,
                    model,
                    input_tokens,
                    output_tokens
                )

                # Prepare MongoDB document
                log_data = {
                    "provider": provider_name,
                    "model": model,
                    "prompt": prompt,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "latency_ms": latency,
                    "cost": cost,
                    "attempts": attempt,
                    "status": "success",
                    "error": None
                }

                # Save successful request
                try:
                    self.logger.log_call(log_data)

                except Exception as log_error:
                    print("MongoDB logging error:", log_error)

                return result["response"]

            except Exception as error:

                last_error = error

                # Identify HTTP status code
                status_code = getattr(error, "status_code", None)

                if status_code is None:
                    status_code = getattr(error, "code", None)

                # Identify retryable errors
                retryable = (
                    isinstance(
                        error,
                        (
                            requests.exceptions.Timeout,
                            requests.exceptions.ConnectionError
                        )
                    )
                    or status_code == 429
                    or (
                        isinstance(status_code, int)
                        and status_code >= 500
                    )
                    or (
                        isinstance(error, RuntimeError)
                        and "Ollama Error 5" in str(error)
                    )
                )

                # Stop retrying if error is permanent
                if not retryable or attempt == max_attempts:
                    break

                wait_time = 2 ** (attempt - 1)

                print(
                    f"Attempt {attempt} failed. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

        # --------------------------------
        # 5. Log Failed Requests
        # --------------------------------

        latency = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        failed_log = {
            "provider": type(self.provider).__name__,
            "model": self.provider.model,
            "prompt": prompt,
            "latency_ms": latency,
            "attempts": attempt,
            "status": "failed",
            "error": str(last_error)
        }

        try:
            self.logger.log_call(failed_log)

        except Exception as log_error:
            print("MongoDB logging error:", log_error)

        raise last_error
