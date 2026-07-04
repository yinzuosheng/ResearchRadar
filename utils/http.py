from typing import Any

import requests

from utils.logger import logger


def get(url: str, headers: dict | None = None, params: dict | None = None, timeout: int = 20) -> requests.Response:
    logger.info("http get %s", url)
    return requests.get(url, headers=headers, params=params, timeout=timeout)


def post_json(url: str, payload: dict[str, Any], headers: dict | None = None, timeout: int = 20) -> requests.Response:
    logger.info("http post %s", url)
    return requests.post(url, json=payload, headers=headers, timeout=timeout)
