"""
voice/stt.py — Real-time speech-to-text using Deepgram Nova-2 (SDK v2.12.0).

How it works:
  1. The frontend sends raw PCM audio chunks over a WebSocket.
  2. We forward those chunks to Deepgram's streaming WebSocket.
  3. Deepgram fires transcript events:
       - is_final=False  → interim result, used for barge-in detection
       - speech_final=True → user stopped speaking, trigger the agent
  4. We pass back the final transcript + detected language.

Deepgram Nova-2 with language="multi" handles Hindi, Tamil, and English.
detected_language from Deepgram is used as the primary language hint;
langdetect is the fallback.
"""

import asyncio
from dataclasses import dataclass
from typing import Callable, Awaitable

from deepgram import Deepgram
from loguru import logger

from config import settings
from agent.lang_detect import detect_language


@dataclass
class TranscriptResult:
    text: str
    language: str
    is_final: bool
    confidence: float


class STTClient:
    def __init__(self):
        self.client = Deepgram(settings.deepgram_api_key)

    def session(self, on_interim, on_final):
        return _STTSession(self.client, on_interim, on_final)


class _STTSession:
    def __init__(self, client, on_interim, on_final):
        self._client = client
        self._on_interim = on_interim
        self._on_final = on_final
        self._connection = None
        self._current_text = ""
        self._detected_language = "en"

    async def __aenter__(self):
        self._connection = await self._client.transcription.live({
            "model": "nova-2",
            "language": "multi",    # enables Hindi, Tamil, and English detection
            "encoding": "linear16",
            "sample_rate": 16000,
            "channels": 1,
            "interim_results": True,
            "endpointing": True,
            "smart_format": True,
        })

        self._connection.registerHandler(
            self._connection.event.TRANSCRIPT_RECEIVED,
            self._on_transcript
        )
        self._connection.registerHandler(
            self._connection.event.CLOSE,
            lambda c: logger.info("Deepgram closed")
        )
        self._connection.registerHandler(
            self._connection.event.ERROR,
            lambda e: logger.error(f"Deepgram error: {e}")
        )

        logger.info("Deepgram STT session started")
        return self._send_audio

    async def _send_audio(self, chunk: bytes):
        if self._connection:
            self._connection.send(chunk)

    async def _on_transcript(self, transcript):
        try:
            if not transcript.get("is_final") and not transcript.get("speech_final"):
                # Interim result
                text = transcript.get("channel", {}).get(
                    "alternatives", [{}])[0].get("transcript", "").strip()
                if text:
                    result = TranscriptResult(
                        text=text,
                        language=self._detected_language,
                        is_final=False,
                        confidence=0.5,
                    )
                    await self._on_interim(result)
                return

            channel = transcript.get("channel", {})
            alt = channel.get("alternatives", [{}])[0]
            text = alt.get("transcript", "").strip()
            if not text:
                return

            # Use Deepgram's own detected_language as the primary hint;
            # langdetect is the fallback (see lang_detect.detect_language priority).
            deepgram_lang = channel.get("detected_language") or alt.get("detected_language")
            language = detect_language(text, deepgram_hint=deepgram_lang)
            self._detected_language = language
            self._current_text = text

            logger.debug(f"STT final [{language}]: {text}")

            result = TranscriptResult(
                text=text,
                language=language,
                is_final=True,
                confidence=alt.get("confidence", 1.0),
            )
            await self._on_final(result)
            self._current_text = ""

        except Exception as e:
            logger.error(f"Transcript error: {e}")

    async def __aexit__(self, *args):
        if self._connection:
            await self._connection.finish()
            logger.info("Deepgram STT session closed")