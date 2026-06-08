"""
NER module — extract named entities from text using spaCy.
"""

import logging
from collections import defaultdict
from functools import lru_cache

import spacy

log = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load spaCy model once and cache it (loading is slow)."""
    log.info(f"Loading spaCy model: {model_name}")
    return spacy.load(model_name)


def extract_entities(text: str, model: str = "en_core_web_sm") -> dict:
    """
    Extract named entities from text.

    Args:
        text: input text (typically OCR output).
        model: spaCy model name. 'en_core_web_sm' for English,
               'de_core_news_sm' for German.

    Returns:
        Dict keyed by entity label (PERSON, ORG, DATE, ...) with lists of
        unique entity strings. Empty dict on failure or empty input.
    """
    if not text or not text.strip():
        return {}

    try:
        nlp = _load_model(model)
        doc = nlp(text)

        grouped = defaultdict(list)
        for ent in doc.ents:
            value = ent.text.strip()
            if value and value not in grouped[ent.label_]:
                grouped[ent.label_].append(value)

        return dict(grouped)
    except Exception as e:
        log.error(f"NER failed: {e}")
        return {}
