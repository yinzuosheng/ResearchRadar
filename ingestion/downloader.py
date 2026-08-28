"""Bounded, validated downloads of legally resolved open-access PDFs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import requests
from pydantic import BaseModel

from domain.models import PaperCandidate
from providers.base import FullTextLocation
from providers.registry import ProviderRegistry


MAX_PDF_BYTES = 50 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 64 * 1024


class DownloadResult(BaseModel):
    success: bool
    path: Path | None = None
    location: FullTextLocation | None = None
    error_code: str | None = None


class PdfDownloader:
    def __init__(
        self,
        registry: ProviderRegistry,
        *,
        http_get: Callable[..., object] = requests.get,
        max_bytes: int = MAX_PDF_BYTES,
        timeout: int = 30,
    ) -> None:
        self.registry = registry
        self.http_get = http_get
        self.max_bytes = max_bytes
        self.timeout = timeout

    def download(self, candidate: PaperCandidate, target_path: Path) -> DownloadResult:
        try:
            locations = [
                location
                for location in self.registry.resolve_full_text(candidate)
                if location.is_oa
            ]
        except Exception:
            return DownloadResult(
                success=False,
                error_code="full_text_resolution_failed",
            )
        if not locations:
            return DownloadResult(success=False, error_code="no_open_full_text")

        final_path = Path(target_path).with_suffix(".pdf")
        part_path = Path(target_path).with_suffix(".part")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        failure_codes: list[str] = []

        attempted_urls: set[str] = set()
        for pass_index in range(2):
            for location in locations:
                if location.url in attempted_urls:
                    continue
                attempted_urls.add(location.url)
                part_path.unlink(missing_ok=True)
                response = None
                try:
                    response = self.http_get(
                        location.url,
                        stream=True,
                        timeout=self.timeout,
                        headers={"Accept": "application/pdf"},
                    )
                    response.raise_for_status()
                    content_type = _response_header(response.headers, "content-type") or ""
                    media_type = content_type.split(";", 1)[0].strip().lower()
                    if media_type not in {"", "application/pdf", "application/octet-stream"}:
                        failure_codes.append("invalid_pdf")
                        continue

                    content_length = _response_header(response.headers, "content-length")
                    if content_length is not None:
                        try:
                            if int(content_length) > self.max_bytes:
                                failure_codes.append("pdf_too_large")
                                continue
                        except (TypeError, ValueError):
                            pass

                    total = 0
                    prefix = bytearray()
                    with part_path.open("wb") as handle:
                        for block in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                            if not block:
                                continue
                            total += len(block)
                            if total > self.max_bytes:
                                raise _DownloadValidationError("pdf_too_large")
                            if len(prefix) < 5:
                                prefix.extend(block[: 5 - len(prefix)])
                            handle.write(block)

                    if bytes(prefix) != b"%PDF-":
                        failure_codes.append("invalid_pdf")
                        continue
                    part_path.replace(final_path)
                    return DownloadResult(
                        success=True,
                        path=final_path,
                        location=location,
                    )
                except _DownloadValidationError as error:
                    failure_codes.append(error.code)
                except Exception:
                    failure_codes.append("pdf_download_failed")
                finally:
                    if response is not None:
                        close = getattr(response, "close", None)
                        if close is not None:
                            close()
                    part_path.unlink(missing_ok=True)

            if pass_index == 0:
                fallback_resolver = getattr(
                    self.registry, "resolve_fallback_full_text", None
                )
                if fallback_resolver is None:
                    break
                try:
                    locations = [
                        location
                        for location in fallback_resolver(candidate)
                        if location.is_oa
                    ]
                except Exception:
                    break

        error_code = _preferred_error_code(failure_codes)
        return DownloadResult(success=False, error_code=error_code)

    def reuse_existing(self, target_path: Path) -> DownloadResult | None:
        """Return a validated completed checkpoint without resolving or downloading."""
        final_path = Path(target_path).with_suffix(".pdf")
        try:
            if not final_path.is_file():
                return None
            size = final_path.stat().st_size
            if size <= 0 or size > self.max_bytes:
                return None
            with final_path.open("rb") as handle:
                if handle.read(5) != b"%PDF-":
                    return None
        except OSError:
            return None
        return DownloadResult(success=True, path=final_path)


class _DownloadValidationError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _preferred_error_code(codes: list[str]) -> str:
    for code in ("pdf_too_large", "invalid_pdf", "pdf_download_failed"):
        if code in codes:
            return code
    return "no_open_full_text"


def _response_header(headers, name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None
