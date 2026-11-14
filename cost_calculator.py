
def calculate_cost(provider, model, input_tokens, output_tokens):

    # Ollama is locally hosted.
    if provider == "OllamaProvider":
        return 0.0

    # Validate token counts.
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")

    # Gemini pricing must be configured using verified rates.
    pricing = {}

    rates = pricing.get(model)

    if rates is None:
        return None

    input_cost = (
        input_tokens * rates["input"]
    ) / 1_000_000

    output_cost = (
        output_tokens * rates["output"]
    ) / 1_000_000

    return round(input_cost + output_cost, 10)
