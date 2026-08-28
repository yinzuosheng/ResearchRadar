import os

from langchain_openai import ChatOpenAI


def build_chat_model() -> ChatOpenAI:
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
    timeout = float(os.getenv("MODEL_REQUEST_TIMEOUT", "45"))
    max_retries = int(os.getenv("MODEL_MAX_RETRIES", "1"))
    client_options = {}
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        client_options["base_url"] = base_url
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        **client_options,
    )
