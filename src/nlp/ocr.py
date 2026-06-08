"""
OCR module — extract raw text from a document image.

Backends (selected via OCR_BACKEND env var):
- 'tesseract' (default, local): pytesseract, free, offline.
- 'textract' (production, Lambda): AWS Textract, managed.
"""

import io
import logging
import os

from PIL import Image

log = logging.getLogger(__name__)

BACKEND = os.getenv("OCR_BACKEND", "tesseract").lower()


def _ocr_tesseract(image: Image.Image, lang: str) -> str:
    import pytesseract  # imported only when used
    return pytesseract.image_to_string(image, lang=lang).strip()


def _ocr_textract(image: Image.Image, lang: str) -> str:
    import boto3  # imported only when used
    client = boto3.client("textract")

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    response = client.detect_document_text(Document={"Bytes": buf.getvalue()})

    lines = [b["Text"] for b in response.get("Blocks", []) if b["BlockType"] == "LINE"]
    return "\n".join(lines).strip()


def extract_text(image: Image.Image, lang: str = "eng") -> str:
    """Run OCR on a PIL image. Backend chosen via OCR_BACKEND env var."""
    try:
        if BACKEND == "textract":
            return _ocr_textract(image, lang)
        return _ocr_tesseract(image, lang)
    except Exception as e:
        log.error(f"OCR failed ({BACKEND}): {e}")
        return ""
