from __future__ import annotations

import model.factory as factory


class FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_build_chat_model_passes_bounded_request_timeout(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("MODEL_TEMPERATURE", "0.4")
    monkeypatch.setenv("MODEL_REQUEST_TIMEOUT", "20")
    monkeypatch.setenv("MODEL_MAX_RETRIES", "1")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(factory, "ChatOpenAI", FakeChatOpenAI)

    model = factory.build_chat_model()

    assert model.kwargs == {
        "model": "test-model",
        "temperature": 0.4,
        "timeout": 20.0,
        "max_retries": 1,
    }


def test_build_chat_model_uses_safe_defaults(monkeypatch):
    monkeypatch.delenv("MODEL_REQUEST_TIMEOUT", raising=False)
    monkeypatch.delenv("MODEL_MAX_RETRIES", raising=False)
    monkeypatch.setattr(factory, "ChatOpenAI", FakeChatOpenAI)

    model = factory.build_chat_model()

    assert model.kwargs["timeout"] == 45.0
    assert model.kwargs["max_retries"] == 1


def test_build_chat_model_passes_openai_compatible_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setattr(factory, "ChatOpenAI", FakeChatOpenAI)

    model = factory.build_chat_model()

    assert model.kwargs["base_url"] == "https://gateway.example/v1"
