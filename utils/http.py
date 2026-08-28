from typing import Any
from urllib.parse import urlparse

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from utils.config import load_tools_config
from utils.logger import logger


class RetryableHttpError(Exception):
    """Raised when an HTTP response should be retried."""

    def __init__(self, status_code: int, retry_after: float | None = None):
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(f"Retryable HTTP status: {status_code}")


def _log_retry(retry_state) -> None:
    error = retry_state.outcome.exception()
    if isinstance(error, RetryableHttpError):
        url = retry_state.args[0] if retry_state.args else retry_state.kwargs.get("url", "")
        logger.warning(
            "retrying HTTP request host=%s status=%s",
            urlparse(url).hostname,
            error.status_code,
        )


_HTTP_CONFIG = load_tools_config().get("http", {})
_RETRY_ATTEMPTS = int(_HTTP_CONFIG.get("retry_attempts", 3))


def _retry_wait(retry_state) -> float:
    error = retry_state.outcome.exception()
    if isinstance(error, RetryableHttpError) and error.retry_after is not None:
        return min(max(error.retry_after, 0.0), 60.0)
    if isinstance(error, RetryableHttpError) and error.status_code == 429:
        # Providers such as Semantic Scholar enforce a cumulative one-request
        # per-second quota and may omit Retry-After.  The generic exponential
        # delay (0.1s/0.2s) would immediately violate that contract.
        return max(1.1, min(0.1 * (2 ** max(retry_state.attempt_number - 1, 0)), 2.0))
    return min(0.1 * (2 ** max(retry_state.attempt_number - 1, 0)), 2.0)


@retry(
    retry=retry_if_exception_type(RetryableHttpError),
    stop=stop_after_attempt(_RETRY_ATTEMPTS),
    wait=_retry_wait,
    before_sleep=_log_retry,
    reraise=True,
)
def get_with_retry(url, *, headers=None, params=None, timeout=None):
    """GET a URL, retrying rate-limit and server responses."""
    cfg = load_tools_config().get("http", {})
    resolved_timeout = timeout or int(cfg.get("timeout_seconds", 30))
    response = requests.get(url, headers=headers, params=params, timeout=resolved_timeout)
    if response.status_code == 429 or response.status_code >= 500:
        retry_after = None
        for key, value in getattr(response, "headers", {}).items():
            if key.lower() == "retry-after":
                try:
                    retry_after = float(value)
                except (TypeError, ValueError):
                    retry_after = None
                break
        raise RetryableHttpError(response.status_code, retry_after=retry_after)
    response.raise_for_status()
    return response


def get_once(url, *, headers=None, params=None, timeout=None):
    """Perform one bounded GET without automatic retries.

    This is used for providers with a cumulative quota where retrying a
    rejected request would create more traffic than the caller intended.
    """
    cfg = load_tools_config().get("http", {})
    resolved_timeout = timeout or int(cfg.get("timeout_seconds", 30))
    response = requests.get(url, headers=headers, params=params, timeout=resolved_timeout)
    if response.status_code == 429 or response.status_code >= 500:
        retry_after = None
        for key, value in getattr(response, "headers", {}).items():
            if key.lower() == "retry-after":
                try:
                    retry_after = float(value)
                except (TypeError, ValueError):
                    retry_after = None
                break
        raise RetryableHttpError(response.status_code, retry_after=retry_after)
    response.raise_for_status()
    return response


def get(url: str, headers: dict | None = None, params: dict | None = None, timeout: int = 20) -> requests.Response:
    logger.info("http get %s", url)
    return requests.get(url, headers=headers, params=params, timeout=timeout)


def post_json(url: str, payload: dict[str, Any], headers: dict | None = None, timeout: int = 20) -> requests.Response:
    logger.info("http post %s", url)
    return requests.post(url, json=payload, headers=headers, timeout=timeout)
