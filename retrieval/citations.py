"""Deterministic validation for model-proposed evidence references."""

from __future__ import annotations

from typing import get_args, get_origin

from domain.models import ExtractedField, PageText, PaperProfile


class UnsupportedProfileFieldError(TypeError):
    """Raised when a profile field has no deliberate validation rule."""


def _normalized_text(text: str) -> str:
    return " ".join(text.split())


def _validated_field(field: ExtractedField, page_text: dict[int, str]) -> ExtractedField:
    if field.value is None or not field.value.strip():
        return ExtractedField()
    surviving = [
        reference.model_copy(deep=True)
        for reference in field.evidence
        if reference.page_number in page_text
        and bool(_normalized_text(reference.quote))
        and _normalized_text(reference.quote) in page_text[reference.page_number]
    ]
    if not surviving:
        return ExtractedField()
    return field.model_copy(update={"evidence": surviving}, deep=True)


def validate_evidence(profile: PaperProfile, pages: list[PageText]) -> PaperProfile:
    """Return a copy containing only fields supported by supplied page text."""
    page_text = {page.page_number: _normalized_text(page.text) for page in pages}
    validated: dict[str, object] = {}
    profile_type = type(profile)
    for name, model_field in profile_type.model_fields.items():
        value = getattr(profile, name)
        annotation = model_field.annotation
        if annotation is ExtractedField:
            if not isinstance(value, ExtractedField):
                raise UnsupportedProfileFieldError(f"unsupported profile field: {name}")
            validated[name] = _validated_field(value, page_text)
            continue
        if get_origin(annotation) is list and get_args(annotation) == (ExtractedField,):
            if not isinstance(value, list) or not all(
                isinstance(item, ExtractedField) for item in value
            ):
                raise UnsupportedProfileFieldError(f"unsupported profile field: {name}")
            validated[name] = [
                supported
                for item in value
                if (supported := _validated_field(item, page_text)).value is not None
            ]
            continue
        raise UnsupportedProfileFieldError(f"unsupported profile field: {name}")
    return profile_type.model_validate(validated)
