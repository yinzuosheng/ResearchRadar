"""Sanitized, dependency-injected provider health aggregation."""

from __future__ import annotations

import time
from typing import Callable, Mapping


PROVIDERS = ("openalex", "core", "unpaywall", "crossref")
STATUSES = {"ok", "missing_configuration", "unreachable", "http_error"}


def _safe_integer(value):
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def run_provider_health(
    probes: Mapping[str, Callable[[], object]],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> list[dict[str, object]]:
    """Run exactly one independent probe per provider and emit safe fields only."""
    results: list[dict[str, object]] = []
    provider_order = (*PROVIDERS, *tuple(name for name in probes if name not in PROVIDERS))
    for provider in provider_order:
        started = clock()
        try:
            payload = probes[provider]()
            payload = payload if isinstance(payload, dict) else {}
            http_status = _safe_integer(payload.get("status_code"))
            explicit = payload.get("status")
            if explicit in STATUSES:
                status = explicit
            elif http_status is not None and 200 <= http_status < 400:
                status = "ok"
            elif http_status is not None:
                status = "http_error"
            else:
                status = "unreachable"
        except Exception as error:
            payload = {}
            response = getattr(error, "response", None)
            http_status = _safe_integer(getattr(response, "status_code", None))
            status = "http_error" if http_status is not None else "unreachable"
        elapsed = max(0.0, clock() - started)
        item: dict[str, object] = {
            "provider": provider,
            "status": status,
            "latency_ms": min(int(round(elapsed * 1000)), 120_000),
        }
        if http_status is not None:
            item["http_status"] = http_status
        quota = _safe_integer(payload.get("remaining_quota"))
        if quota is not None:
            item["remaining_quota"] = quota
        results.append(item)
    return results
