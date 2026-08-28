from types import SimpleNamespace

import pytest

from utils import http


class FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code
        self.raise_for_status_called = False

    def raise_for_status(self) -> None:
        self.raise_for_status_called = True
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_get_with_retry_retries_rate_limit_and_server_error(monkeypatch):
    responses = [FakeResponse(429), FakeResponse(503), FakeResponse(200)]
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return responses.pop(0)

    monkeypatch.setattr(http.requests, "get", fake_get)

    result = http.get_with_retry("https://api.example.org/works", timeout=1)

    assert result.status_code == 200
    assert result.raise_for_status_called is True
    assert len(calls) == 3


def test_get_with_retry_raises_non_retryable_client_error(monkeypatch):
    response = FakeResponse(404)
    monkeypatch.setattr(http.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="HTTP 404"):
        http.get_with_retry("https://api.example.org/works", timeout=1)

    assert response.raise_for_status_called is True


def test_get_with_retry_accepts_keyword_url_when_retrying(monkeypatch):
    responses = [FakeResponse(429), FakeResponse(200)]

    monkeypatch.setattr(http.requests, "get", lambda *args, **kwargs: responses.pop(0))

    result = http.get_with_retry(url="https://api.example.org/works?api_key=secret", timeout=1)

    assert result.status_code == 200


def test_get_once_does_not_retry_rate_limited_response(monkeypatch):
    calls = []

    class Response:
        status_code = 429
        headers = {}

        def raise_for_status(self):
            raise RuntimeError("HTTP 429")

    def fake_get(*args, **kwargs):
        calls.append(1)
        return Response()

    monkeypatch.setattr(http.requests, "get", fake_get)

    with pytest.raises(http.RetryableHttpError):
        http.get_once("https://api.example.org/works", timeout=1)

    assert len(calls) == 1
