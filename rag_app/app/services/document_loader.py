"""
PDF loading and text extraction using PyMuPDF (fitz).

Kept isolated so that adding support for other formats (docx, txt, html)
later only means adding a new function here, not touching the API layer.
"""
import logging
import re

import pymupdf as fitz  # PyMuPDF (the 'fitz' name is now deprecated in favor of 'pymupdf')

logger = logging.getLogger(__name__)


class DocumentLoadError(Exception):
    """Raised when a document cannot be read or contains no usable text."""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF file's bytes."""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            pages_text = [page.get_text() for page in pdf]
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to open/parse PDF: %s", exc)
        raise DocumentLoadError("The uploaded file is not a valid or readable PDF.") from exc

    text = "\n".join(pages_text)

    if not text.strip():
        raise DocumentLoadError(
            "No extractable text was found in this PDF. It may be a scanned "
            "image without OCR, which this version does not support."
        )

    return clean_text(text)


def clean_text(text: str) -> str:
    """Basic whitespace / artifact cleanup. Keep this simple on purpose."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
