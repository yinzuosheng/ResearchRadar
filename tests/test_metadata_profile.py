from domain.models import PaperCandidate
from ingestion.metadata_profile import MetadataProfileExtractor
from storage.database import ResearchDatabase
from workflows.metadata_profiles import MetadataProfileService


def candidate():
    return PaperCandidate(
        source="openalex",
        source_id="W1",
        title="Sentinel-2 chlorophyll-a prediction with random forest",
        abstract="We estimate water quality and turbidity using Sentinel-2 imagery and an LSTM model.",
    )


def test_metadata_profile_extracts_targets_sensors_models_with_abstract_evidence():
    profile = MetadataProfileExtractor().extract(candidate())
    assert profile.prediction_target.value.casefold() in {"chlorophyll-a", "water quality", "turbidity"}
    assert {field.value.casefold() for field in profile.sensors} >= {"sentinel-2"}
    assert {field.value.casefold() for field in profile.models} >= {"random forest", "lstm"}
    assert profile.models[0].evidence[0].page_number == 0


def test_metadata_profile_service_is_idempotent_and_does_not_change_paper_status(tmp_path):
    database = ResearchDatabase(tmp_path / "research.db")
    record = database.upsert_candidate(candidate())
    first = MetadataProfileService(database).run(limit=10)
    second = MetadataProfileService(database).run(limit=10)
    assert first["profiled"] == 1
    assert second["profiled"] == 0
    assert database.get_paper(record.paper_id).status == "discovered"
