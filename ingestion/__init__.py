"""Page-preserving research-paper ingestion."""

from ingestion.downloader import DownloadResult, PdfDownloader
from ingestion.pdf_parser import PdfParser
from ingestion.pipeline import ResearchIngestor

__all__ = ["DownloadResult", "PdfDownloader", "PdfParser", "ResearchIngestor"]
