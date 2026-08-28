"""Deterministic bilingual query expansion and bounded intent classification."""

from __future__ import annotations

import re
from dataclasses import dataclass


TERM_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("叶绿素a", "叶绿素 a", "叶绿素", "chla", "chl-a"), ("chlorophyll-a", "chlorophyll a", "Chl-a")),
    (("水质", "水质参数"), ("water quality", "water quality parameter")),
    (("遥感", "卫星影像", "卫星遥感"), ("remote sensing", "satellite imagery")),
    (("浊度",), ("turbidity",)),
    (("总悬浮物", "悬浮物", "tss"), ("total suspended solids", "TSS")),
    (("蓝藻", "藻华"), ("cyanobacteria", "algal bloom")),
    (("传感器",), ("sensor", "Sentinel-2", "Landsat", "MODIS")),
    (("湖泊", "湖水", "湖区"), ("lake", "lake water")),
    (("反演", "估算", "反推", "检索"), ("retrieval", "estimation")),
    (("预测", "预报"), ("prediction", "forecasting")),
)

_PLAN_MARKERS = (
    "如何", "怎么", "从哪里开始", "研究路线", "研究方案", "实验设计", "基线", "下一步",
    "how to", "how can", "research plan", "experiment design", "baseline", "next step",
)
_CONCEPT_MARKERS = (
    "是什么", "什么是", "定义", "含义", "解释", "为什么叫",
    "what is", "define", "definition", "meaning of", "explain", "why is",
)
_DOMAIN_MARKERS = (
    "叶绿素", "水质", "遥感", "卫星", "传感器", "论文", "文献", "预测", "浊度", "藻华",
    "chlorophyll", "water quality", "remote sensing", "sensor", "paper", "turbidity",
    "sentinel-2", "landsat", "modis",
)

_FACT_MARKERS = ("哪种", "哪些", "多少", "使用了什么", "结果", "指标", "结论", "对比", "according", "which", "what", "how many")
_RETRIEVAL_TASK_MARKERS = ("预测", "反演", "估算", "识别", "监测", "retrieval", "prediction", "estimation")


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    confidence: float
    normalized_query: str
    queries: tuple[str, ...]
    needs_local_evidence: bool
    clarification_needed: bool = False
    keyword_queries: tuple[str, ...] = ()
    semantic_queries: tuple[str, ...] = ()
    metadata_queries: tuple[str, ...] = ()
    abbreviation_expansions: tuple[str, ...] = ()


def expand_query(query: str, *, max_queries: int = 6) -> list[str]:
    """Return the original query plus deterministic bilingual variants."""
    normalized = " ".join(str(query).split())
    if not normalized:
        return []
    variants = [normalized]
    detected: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for aliases, english in TERM_GROUPS:
        if any(alias.casefold() in normalized.casefold() for alias in aliases):
            detected.append((aliases, english))
    # Put the all-terms bilingual query first. It preserves the user's
    # constraints (for example, sensor + prediction) instead of allowing a
    # broad single-term variant to dominate the top-k budget.
    if len(detected) > 1:
        candidate = normalized
        for aliases, english in detected:
            for alias in aliases:
                candidate = re.sub(re.escape(alias), f" {english[0]} ", candidate, flags=re.IGNORECASE)
        candidate = " ".join(candidate.split())
        if candidate not in variants:
            variants.append(candidate)
    for aliases, english in detected:
        candidate = normalized
        for alias in aliases:
            candidate = re.sub(re.escape(alias), f" {english[0]} ", candidate, flags=re.IGNORECASE)
        candidate = " ".join(candidate.split())
        if candidate not in variants:
            variants.append(candidate)
    return variants[:max_queries]


def classify_question(query: str) -> str:
    """Classify only the interaction boundary; tools still validate evidence."""
    text = " ".join(str(query).split()).casefold()
    if not text:
        return "general_chat"
    if any(marker in text for marker in _PLAN_MARKERS) and any(
        marker.casefold() in text for marker in _DOMAIN_MARKERS
    ):
        return "research_plan"
    if any(marker in text for marker in _CONCEPT_MARKERS) and any(
        marker.casefold() in text for marker in _DOMAIN_MARKERS
    ):
        return "concept_explanation"
    if any(marker.casefold() in text for marker in _DOMAIN_MARKERS):
        return "evidence_qa"
    return "general_chat"


def plan_query(query: str) -> QueryPlan:
    """Create a bounded semantic plan without an extra model call."""
    normalized = " ".join(str(query).split())
    intent = classify_question(normalized)
    text = normalized.casefold()
    variants = expand_query(normalized)
    if intent == "concept_explanation":
        needs_local = False
    else:
        needs_local = intent in {"evidence_qa", "research_plan"}
    if intent == "research_plan" and any(marker in text for marker in _RETRIEVAL_TASK_MARKERS):
        variants.extend([
            f"{normalized} retrieval or prediction task",
            f"{normalized} baseline experiment validation",
        ])
    elif intent == "evidence_qa" and any(marker in text for marker in _FACT_MARKERS):
        variants.append(f"{normalized} method dataset metric evidence")
    deduped = tuple(dict.fromkeys(item for item in variants if item.strip()))[:6]
    keyword_queries = tuple(dict.fromkeys((normalized, *deduped)))[:4]
    semantic_queries = tuple(
        dict.fromkeys(
            (
                normalized,
                *(
                    item
                    for item in deduped
                    if item != normalized and any(char.isascii() for char in item)
                ),
            )
        )
    )[:4]
    metadata_queries = tuple(dict.fromkeys((normalized, *semantic_queries)))[:3]
    abbreviations: list[str] = []
    lowered = normalized.casefold()
    for aliases, english in TERM_GROUPS:
        if any(alias.casefold() in {"chla", "chl-a", "tss"} and alias.casefold() in lowered for alias in aliases):
            abbreviations.extend(english)
    abbreviation_expansions = tuple(dict.fromkeys(abbreviations))
    confidence = 0.95 if intent != "general_chat" else 0.9
    clarification = len(normalized) < 3
    return QueryPlan(
        intent=intent,
        confidence=confidence,
        normalized_query=normalized,
        queries=deduped,
        needs_local_evidence=needs_local,
        clarification_needed=clarification,
        keyword_queries=keyword_queries,
        semantic_queries=semantic_queries,
        metadata_queries=metadata_queries,
        abbreviation_expansions=abbreviation_expansions,
    )


__all__ = ["QueryPlan", "classify_question", "expand_query", "plan_query"]
