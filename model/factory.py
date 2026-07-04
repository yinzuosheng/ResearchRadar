import os

from langchain_openai import ChatOpenAI


def build_chat_model() -> ChatOpenAI:
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
    return ChatOpenAI(model=model_name, temperature=temperature)


chat_model = build_chat_model()
