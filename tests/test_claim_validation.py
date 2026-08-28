from domain.models import AnswerClaim, EvidenceChunk
from workflows.qa import _validate_claims


def _chunk(chunk_id, paper_id, text):
    return EvidenceChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        title=paper_id,
        page_number=1,
        text=text,
    )


def _citation(chunk_id, paper_id, quote):
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "title": paper_id,
        "page_number": 1,
        "quote": quote,
    }


def test_direct_claim_requires_quote_from_same_chunk():
    chunks = [_chunk("c1", "p1", "Sentinel-2 supports chlorophyll prediction.")]
    claims = [
        AnswerClaim(
            text="Sentinel-2 supports chlorophyll prediction.",
            kind="direct",
            citations=[
                _citation(
                    "c1",
                    "wrong",
                    "Sentinel-2 supports chlorophyll prediction.",
                )
            ],
        )
    ]

    assert _validate_claims(claims, chunks) == []


def test_synthesis_claim_requires_two_distinct_papers():
    chunks = [
        _chunk("c1", "p1", "Random forest is used."),
        _chunk("c2", "p1", "SVM is used."),
    ]
    claims = [
        AnswerClaim(
            text="Both papers compare models.",
            kind="synthesis",
            citations=[
                _citation("c1", "p1", "Random forest is used."),
                _citation("c2", "p1", "SVM is used."),
            ],
        )
    ]

    assert _validate_claims(claims, chunks) == []


def test_invalid_claim_is_removed_without_removing_valid_claim():
    chunks = [
        _chunk("c1", "p1", "Valid evidence."),
        _chunk("c2", "p2", "Other evidence."),
    ]
    claims = [
        AnswerClaim(text="unsupported", kind="direct", citations=[]),
        AnswerClaim(
            text="supported",
            kind="direct",
            citations=[_citation("c1", "p1", "Valid evidence.")],
        ),
    ]

    result = _validate_claims(claims, chunks)

    assert [claim.text for claim in result] == ["supported"]
    assert result[0].citations[0].paper_id == "p1"


def test_no_surviving_claim_returns_evidence_insufficient():
    claims = [AnswerClaim(text="unsupported", kind="direct", citations=[])]

    assert _validate_claims(claims, []) == []
