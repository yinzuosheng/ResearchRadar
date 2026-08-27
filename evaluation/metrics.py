"""Pure evaluation metrics with explicit boundary validation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def _validated_ids(values: object) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError("evaluation_metric_invalid")
    items = list(values)
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError("evaluation_metric_invalid")
    return items


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("evaluation_metric_invalid")
    ranked = _validated_ids(ranked_ids)
    relevant = set(_validated_ids(relevant_ids))
    if not relevant:
        return 0.0
    retrieved = set(ranked[:k])
    return len(retrieved & relevant) / len(relevant)


def evidence_group_recall_at_k(
    ranked_ids: Sequence[str],
    relevant_groups: Iterable[Iterable[str]],
    k: int,
) -> float:
    """Measure required evidence groups hit by at least one alternative Chunk."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("evaluation_metric_invalid")
    ranked = set(_validated_ids(ranked_ids)[:k])
    if isinstance(relevant_groups, (str, bytes)) or not isinstance(
        relevant_groups, Iterable
    ):
        raise ValueError("evaluation_metric_invalid")
    groups = [set(_validated_ids(group)) for group in relevant_groups]
    if any(not group for group in groups):
        raise ValueError("evaluation_metric_invalid")
    if not groups:
        return 0.0
    return sum(bool(ranked & group) for group in groups) / len(groups)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    ranked = _validated_ids(ranked_ids)
    relevant = set(_validated_ids(relevant_ids))
    if not relevant:
        return 0.0
    for rank, item_id in enumerate(ranked, start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def _ratio(numerator: int, denominator: int) -> float:
    if (
        not isinstance(numerator, int)
        or isinstance(numerator, bool)
        or not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or numerator < 0
        or denominator < 0
        or numerator > denominator
    ):
        raise ValueError("evaluation_metric_invalid")
    return numerator / denominator if denominator else 0.0


def citation_precision(valid: int, total: int) -> float:
    return _ratio(valid, total)


def evidence_coverage(covered: int, total: int) -> float:
    return _ratio(covered, total)


def unsupported_claim_rate(unsupported: int, total: int) -> float:
    return _ratio(unsupported, total)


def answer_level_metrics(rows: Iterable[dict]) -> dict[str, float | int]:
    """Measure citation support from sanitized answer-evaluation rows."""
    items = list(rows)
    if any(not isinstance(row, dict) for row in items):
        raise ValueError("evaluation_metric_invalid")
    valid_citations = total_citations = covered = confident = unsupported = 0
    for row in items:
        relevant = set(_validated_ids(row.get("relevant_chunk_ids", [])))
        claims = row.get("claims", [])
        if not isinstance(claims, list):
            raise ValueError("evaluation_metric_invalid")
        for claim in claims:
            if not isinstance(claim, dict):
                raise ValueError("evaluation_metric_invalid")
            citation_ids = _validated_ids(claim.get("citation_chunk_ids", []))
            total_citations += len(citation_ids)
            valid = len(set(citation_ids) & relevant)
            valid_citations += valid
            covered += int(valid > 0)
            sufficient = claim.get("evidence_sufficient")
            if not isinstance(sufficient, bool):
                raise ValueError("evaluation_metric_invalid")
            if sufficient:
                confident += 1
                unsupported += int(valid == 0)
    return {
        "citation_precision": citation_precision(valid_citations, total_citations),
        "evidence_coverage": evidence_coverage(covered, len(items)),
        "unsupported_claim_rate": unsupported_claim_rate(unsupported, confident),
        "question_count": len(items),
    }


def deterministic_mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
