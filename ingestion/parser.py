"""
Document parser module — wraps Docling for layout-aware PDF/DOCX parsing.

Design Patterns:
    - Factory Pattern: `DocumentParserFactory` selects the right Docling
      configuration based on file type.
    - Strategy Pattern: The parser interface allows future swap to
      alternative parsing backends.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from core.exceptions import ParsingError, UnsupportedFileTypeError
from core.models import ParsedDocument

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


class BaseDocumentParser(ABC):
    """Abstract base for document parsers (Strategy interface)."""

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """Parse a file and return a ParsedDocument."""


class DoclingDocumentParser(BaseDocumentParser):
    """
    Layout-aware document parser using IBM Docling.

    Handles PDF and DOCX files, preserving structure (headings,
    paragraphs, tables, lists) and extracting rich metadata.
    Runs on CPU by default.
    """

    def __init__(self) -> None:
        self._converter = None

    def _get_converter(self):
        """Lazy-initialise the Docling converter (heavy import)."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
                from docling.document_converter import PdfFormatOption
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
                from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

                # Docling Parse without EasyOCR 
                pipeline_options = PdfPipelineOptions()
                # Disable the heavy layout & table neural models
                pipeline_options.do_ocr = False                  # skip OCR 
                # pipeline_options.ocr_options.use_gpu = False     # Set to false for CPU only OCR
                # pipeline_options.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.AUTO)
                pipeline_options.do_table_structure = True       # skip table detection model

                # Lower the page image resolution (default is very high)
                pipeline_options.images_scale = 1.0              # default is often 2.0+
                pipeline_options.generate_page_images = False    # don't render full page images
                pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)

                self._converter = DocumentConverter(format_options={ InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
                logger.info("Docling DocumentConverter initialised")
            except ImportError as exc:
                raise ParsingError(
                    "Docling is not installed. Run: pip install docling"
                ) from exc
        return self._converter

    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a single file and return a ParsedDocument.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to a PDF or DOCX file.

        Returns
        -------
        ParsedDocument
            Contains the raw Docling document, page count, and metadata.

        Raises
        ------
        UnsupportedFileTypeError
            If the file extension is not supported.
        ParsingError
            If Docling fails to convert the file.
        """
        path = Path(file_path).resolve()
        ext = path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
            )

        if not path.exists():
            raise ParsingError(f"File not found: {path}")

        logger.info("Parsing document: %s", path.name)

        try:
            converter = self._get_converter()
            result = converter.convert(str(path))
            doc = result.document

            # Extract basic metadata
            page_count = len(doc.pages) if hasattr(doc, "pages") else 0
            metadata = {
                "source_path": str(path),
                "file_type": ext,
            }

            parsed = ParsedDocument(
                doc_name=path.name,
                file_path=str(path),
                docling_document=doc,
                page_count=page_count,
                metadata=metadata,
            )

            logger.info(
                "Parsed '%s' — %d page(s)",
                path.name,
                page_count,
            )
            return parsed

        except UnsupportedFileTypeError:
            raise
        except Exception as exc:
            raise ParsingError(
                f"Failed to parse '{path.name}': {exc}"
            ) from exc


class DocumentParserFactory:
    """
    Factory for creating the appropriate parser instance.

    Currently all supported types go through DoclingDocumentParser,
    but the factory allows registering specialised parsers per
    extension in the future (e.g., an OCR-optimised PDF parser).
    """

    _parsers: dict[str, type[BaseDocumentParser]] = {
        ".pdf": DoclingDocumentParser,
        ".docx": DoclingDocumentParser,
    }

    @classmethod
    def register(cls, extension: str, parser_class: type[BaseDocumentParser]) -> None:
        """Register a parser for a file extension."""
        cls._parsers[extension.lower()] = parser_class
        logger.info("Registered parser %s for '%s'", parser_class.__name__, extension)

    @classmethod
    def create(cls, file_path: str) -> BaseDocumentParser:
        """
        Return a parser instance suitable for the given file.

        Raises
        ------
        UnsupportedFileTypeError
            If no parser is registered for the file's extension.
        """
        ext = Path(file_path).suffix.lower()
        parser_class = cls._parsers.get(ext)
        if parser_class is None:
            raise UnsupportedFileTypeError(
                f"No parser registered for '{ext}'. "
                f"Supported: {list(cls._parsers.keys())}"
            )
        return parser_class()
