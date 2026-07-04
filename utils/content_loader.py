from io import BytesIO

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from utils.config import load_tools_config
from utils.logger import logger

try:
    import trafilatura
except ImportError:
    trafilatura = None


def load_url_text(url: str) -> str:
    cfg = load_tools_config().get("http", {})
    headers = {"User-Agent": cfg.get("user_agent", "IntelRagAgent/1.0")}
    timeout = int(cfg.get("timeout_seconds", 20))
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "").lower()
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        return _extract_pdf_text(resp.content)

    return _extract_html_text(resp.text)


def load_pdf_text(url: str) -> str:
    cfg = load_tools_config().get("http", {})
    headers = {"User-Agent": cfg.get("user_agent", "IntelRagAgent/1.0")}
    timeout = int(cfg.get("timeout_seconds", 20))
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return _extract_pdf_text(resp.content)


def _extract_pdf_text(payload: bytes) -> str:
    reader = PdfReader(BytesIO(payload))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _extract_html_text(html: str) -> str:
    if trafilatura is not None:
        extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
        if extracted:
            return extracted.strip()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = " ".join(soup.stripped_strings)
    logger.info("html extracted length=%s", len(text))
    return text
