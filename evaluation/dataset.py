"""Bounded JSONL evaluation dataset loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


MAX_ROWS = 1_000
MAX_LINE_BYTES = 65_536
_PLACEHOLDER = "__ANNOTATE_FROM_LOCAL_SEED__"
_PLACEHOLDER_PATTERN = re.compile(
    r"(?i)(?:__annotate|placeholder|\btodo\b|\btbd\b|(?:^|[_-])id[_-]?here(?:$|[_-]))"
)
_SENSITIVE = re.compile(
    r"(?i)(?:[a-z]:[\\/]|(?:^|\s)\\\\|(?:^|\s)/(?:[^/\s]+/)|https?://|"
    r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|authorization|api[_ -]?key|"
    r"password|credential|bearer\s+|sk-[a-z0-9])"
)
_CATEGORY = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class DatasetError(ValueError):
    """Stable, non-echoing dataset validation error."""


@dataclass(frozen=True)
class EvidenceGroup:
    paper_id: str
    chunk_ids: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class EvaluationQuestion:
    question_id: str
    question: str
    evidence_groups: tuple[EvidenceGroup, ...]
    category: str = "uncategorized"
    placeholder: bool = False

    @property
    def relevant_paper_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(group.paper_id for group in self.evidence_groups))

    @property
    def relevant_chunk_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                chunk_id
                for group in self.evidence_groups
                for chunk_id in group.chunk_ids
            )
        )


@dataclass(frozen=True)
class AnswerEvaluationRow:
    question_id: str
    relevant_chunk_ids: tuple[str, ...]
    claims: tuple[dict, ...]


def _stable_id(question: str) -> str:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    return f"derived-{digest}"


def _valid_ids(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        return None
    normalized = tuple(item.strip() for item in value)
    if len(set(normalized)) != len(normalized):
        return None
    return normalized


def _parse_evidence_groups(value: object) -> tuple[EvidenceGroup, ...]:
    if not isinstance(value, list) or not value:
        raise DatasetError("evaluation_dataset_invalid_evidence_group")
    groups: list[EvidenceGroup] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    expected_keys = {"paper_id", "chunk_ids", "rationale"}
    for payload in value:
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise DatasetError("evaluation_dataset_invalid_evidence_group")
        paper_id = payload.get("paper_id")
        rationale = payload.get("rationale")
        chunk_ids = _valid_ids(payload.get("chunk_ids"))
        if (
            not isinstance(paper_id, str)
            or not paper_id.strip()
            or not isinstance(rationale, str)
            or not rationale.strip()
            or chunk_ids is None
            or not chunk_ids
        ):
            raise DatasetError("evaluation_dataset_invalid_evidence_group")
        paper_id = paper_id.strip()
        rationale = rationale.strip()
        if any(not chunk_id.startswith(f"{paper_id}:") for chunk_id in chunk_ids):
            raise DatasetError("evaluation_dataset_invalid_evidence_group")
        group_key = (paper_id, chunk_ids, rationale)
        if group_key in seen:
            raise DatasetError("evaluation_dataset_invalid_evidence_group")
        seen.add(group_key)
        groups.append(
            EvidenceGroup(
                paper_id=paper_id,
                chunk_ids=chunk_ids,
                rationale=rationale,
            )
        )
    return tuple(groups)


def _parse_row(payload: object, *, template_mode: bool) -> EvaluationQuestion:
    if not isinstance(payload, dict):
        raise DatasetError("evaluation_dataset_invalid")
    allowed = {
        "question_id",
        "question",
        "category",
        "evidence_groups",
    }
    if not set(payload).issubset(allowed) or "evidence_groups" not in payload:
        raise DatasetError("evaluation_dataset_invalid")
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise DatasetError("evaluation_dataset_invalid")
    question = question.strip()
    category = payload.get("category", "uncategorized")
    if not isinstance(category, str) or not _CATEGORY.fullmatch(category.strip()):
        raise DatasetError("evaluation_dataset_invalid_category")
    category = category.strip()
    question_id = payload.get("question_id")
    if question_id is None:
        question_id = _stable_id(question)
    if not isinstance(question_id, str) or not question_id.strip():
        raise DatasetError("evaluation_dataset_invalid")
    evidence_groups = _parse_evidence_groups(payload.get("evidence_groups"))
    values = (
        question_id,
        question,
        *(
            value
            for group in evidence_groups
            for value in (group.paper_id, *group.chunk_ids, group.rationale)
        ),
    )
    if any(_SENSITIVE.search(value) for value in values):
        raise DatasetError("evaluation_dataset_sensitive_content")
    placeholder = any(
        value == _PLACEHOLDER or _PLACEHOLDER_PATTERN.search(value)
        for group in evidence_groups
        for value in (group.paper_id, *group.chunk_ids, group.rationale)
    )
    if placeholder and not template_mode:
        raise DatasetError("evaluation_dataset_unannotated")
    return EvaluationQuestion(
        question_id=question_id.strip(),
        question=question,
        evidence_groups=evidence_groups,
        category=category,
        placeholder=placeholder,
    )


def load_questions(path: str | Path, *, template_mode: bool = False) -> list[EvaluationQuestion]:
    """Load a bounded UTF-8 JSONL dataset without echoing rejected contents."""
    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()
    seen_rows: set[tuple[object, ...]] = set()
    try:
        with Path(path).open("rb") as stream:
            for line_number in range(1, MAX_ROWS + 2):
                raw_line = stream.readline(MAX_LINE_BYTES + 1)
                if not raw_line:
                    break
                if line_number > MAX_ROWS or len(raw_line) > MAX_LINE_BYTES:
                    raise DatasetError("evaluation_dataset_too_large")
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeError:
                    raise DatasetError("evaluation_dataset_malformed_json") from None
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except (json.JSONDecodeError, UnicodeError):
                    raise DatasetError("evaluation_dataset_malformed_json") from None
                question = _parse_row(payload, template_mode=template_mode)
                row_key = (
                    question.question,
                    question.evidence_groups,
                )
                if row_key in seen_rows:
                    raise DatasetError("evaluation_dataset_duplicate_row")
                if question.question_id in seen_ids:
                    raise DatasetError("evaluation_dataset_duplicate_question_id")
                seen_rows.add(row_key)
                seen_ids.add(question.question_id)
                questions.append(question)
    except DatasetError:
        raise
    except (OSError, UnicodeError):
        raise DatasetError("evaluation_dataset_unreadable") from None
    if not questions:
        raise DatasetError("evaluation_dataset_empty")
    return questions


def load_answer_rows(path: str | Path) -> list[dict]:
    """Load sanitized answer/citation rows for answer-level evaluation."""
    rows: list[dict] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        raise DatasetError("evaluation_answer_dataset_unreadable") from None
    if len(lines) > MAX_ROWS:
        raise DatasetError("evaluation_answer_dataset_too_large")
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            raise DatasetError("evaluation_answer_dataset_malformed_json") from None
        if not isinstance(payload, dict) or set(payload) != {
            "question_id", "relevant_chunk_ids", "claims"
        }:
            raise DatasetError("evaluation_answer_dataset_invalid")
        question_id = payload.get("question_id")
        relevant = _valid_ids(payload.get("relevant_chunk_ids"))
        claims = payload.get("claims")
        if (
            not isinstance(question_id, str)
            or not question_id.strip()
            or relevant is None
            or not relevant
            or not isinstance(claims, list)
            or not claims
        ):
            raise DatasetError("evaluation_answer_dataset_invalid")
        normalized_claims: list[dict] = []
        for claim in claims:
            if not isinstance(claim, dict) or set(claim) != {
                "text", "evidence_sufficient", "citation_chunk_ids"
            }:
                raise DatasetError("evaluation_answer_dataset_invalid")
            text = claim.get("text")
            citation_ids = _valid_ids(claim.get("citation_chunk_ids"))
            sufficient = claim.get("evidence_sufficient")
            if (
                not isinstance(text, str)
                or not text.strip()
                or citation_ids is None
                or not isinstance(sufficient, bool)
            ):
                raise DatasetError("evaluation_answer_dataset_invalid")
            values = (question_id, text, *relevant, *citation_ids)
            if any(_SENSITIVE.search(value) for value in values):
                raise DatasetError("evaluation_answer_dataset_sensitive_content")
            normalized_claims.append(
                {
                    "text": text.strip(),
                    "evidence_sufficient": sufficient,
                    "citation_chunk_ids": list(citation_ids),
                }
            )
        rows.append(
            {
                "question_id": question_id.strip(),
                "relevant_chunk_ids": list(relevant),
                "claims": normalized_claims,
            }
        )
    if not rows or len({row["question_id"] for row in rows}) != len(rows):
        raise DatasetError("evaluation_answer_dataset_invalid")
    return rows
