"""
agent/lang_detect.py — Detect language from transcript text.

We use two strategies:
  1. Deepgram's detected_language field (fastest, happens during STT)
  2. langdetect library as fallback

The detected language is stored in session memory so it persists for
the whole conversation. For returning patients, their stored language
preference from the DB overrides detection.
"""

from langdetect import detect, DetectorFactory
from loguru import logger

from config import settings

# Make langdetect deterministic
DetectorFactory.seed = 0

# Supported languages
SUPPORTED_LANGUAGES = {"en", "hi", "ta"}

# Map language code → ElevenLabs voice ID
VOICE_MAP = {
    "en": settings.elevenlabs_voice_en,
    "hi": settings.elevenlabs_voice_hi or settings.elevenlabs_voice_en,
    "ta": settings.elevenlabs_voice_ta or settings.elevenlabs_voice_en,
}

# Language names for logging / prompts
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
}


def detect_language(text: str, deepgram_hint: str | None = None) -> str:
    """
    Returns a language code: "en", "hi", or "ta".

    Priority:
      1. deepgram_hint (from Deepgram's detected_language field)
      2. langdetect
      3. Fallback to "en"
    """
    # Deepgram tells us directly — trust it first
    if deepgram_hint and deepgram_hint[:2] in SUPPORTED_LANGUAGES:
        return deepgram_hint[:2]

    if not text or len(text.strip()) < 3:
        return "en"

    try:
        detected = detect(text)
        lang = detected[:2]
        if lang in SUPPORTED_LANGUAGES:
            return lang
    except Exception as e:
        logger.warning(f"langdetect failed: {e}")

    return "en"


def get_voice_id(language: str) -> str:
    """Return the ElevenLabs voice ID for a given language code."""
    return VOICE_MAP.get(language, VOICE_MAP["en"])
