"""Legal open-literature provider adapters."""

from providers.base import DiscoveryProvider, FullTextLocation, MetadataEnricher, OaResolver
from providers.core import CoreProvider
from providers.crossref import CrossrefEnricher
from providers.openalex import OpenAlexProvider
from providers.registry import ProviderRegistry
from providers.unpaywall import UnpaywallResolver

__all__ = [
    "CoreProvider",
    "CrossrefEnricher",
    "DiscoveryProvider",
    "FullTextLocation",
    "MetadataEnricher",
    "OaResolver",
    "OpenAlexProvider",
    "ProviderRegistry",
    "UnpaywallResolver",
]
