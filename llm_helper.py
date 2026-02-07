import os

def get_model():
    # Configure litellm (uses standard env vars like OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER")

    MODEL_ID = os.getenv("MODEL_ID", "gpt-4o-mini")
    if LLM_PROVIDER == "openrouter":
        MODEL = f"openrouter/{MODEL_ID}"
    else:
        MODEL = MODEL_ID

    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"Using model: {MODEL}")
    return MODEL
