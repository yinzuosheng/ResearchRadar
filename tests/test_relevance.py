from domain.models import PaperCandidate
from workflows.relevance import is_relevant, relevance_groups


def paper(title, abstract=""):
    return PaperCandidate(source="openalex", source_id=title, title=title, abstract=abstract)


def test_relevance_requires_multiple_research_dimensions():
    item = paper("Sentinel-2 chlorophyll-a prediction with machine learning")
    assert relevance_groups(item) == {"target", "sensor", "method"}
    assert is_relevant(item, minimum_groups=2)
    assert not is_relevant(paper("Satellite image classification"), minimum_groups=2)
