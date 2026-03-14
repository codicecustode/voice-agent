"""
voice/stt.py — Real-time speech-to-text using Deepgram Nova-2.

How it works:
  1. The frontend sends raw PCM audio chunks over a WebSocket.
  2. We forward those chunks to Deepgram's streaming WebSocket.
  3. Deepgram fires two kinds of events:
       - interim_results: partial transcripts (we use for barge-in detection)
       - UtteranceEnd:    signals the user stopped speaking → we trigger the agent
  4. We pass back the final transcript + detected language.

Deepgram handles Hindi, Tamil, and English natively in the "nova-2" model.
"""

"""
voice/stt.py — Real-time STT using Deepgram SDK v3
"""

import asyncio
from dataclasses import dataclass
from typing import Callable, Awaitable

from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
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
        self.client = DeepgramClient(settings.deepgram_api_key)

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
        options = LiveOptions(
            model="nova-2",
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            interim_results=True,
            utterance_end_ms=800,
            smart_format=True,
        )

        self._connection = self._client.listen.asynclive.v("1")

        self._connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
        self._connection.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)
        self._connection.on(LiveTranscriptionEvents.Error, self._on_error)

        if not await self._connection.start(options):
            raise RuntimeError("Failed to start Deepgram connection")

        logger.info("Deepgram STT session started")
        return self._send_audio

    async def _send_audio(self, chunk: bytes):
        if self._connection:
            await self._connection.send(chunk)

    async def _on_transcript(self, _client, result, **kwargs):
        try:
            alt = result.channel.alternatives[0]
            text = alt.transcript.strip()
            if not text:
                return

            lang_hint = getattr(result, "detected_language", None)
            language = detect_language(text, deepgram_hint=lang_hint)
            self._detected_language = language

            transcript = TranscriptResult(
                text=text,
                language=language,
                is_final=result.is_final,
                confidence=alt.confidence,
            )

            if result.is_final:
                self._current_text = text
                logger.debug(f"STT final [{language}]: {text}")
            else:
                asyncio.create_task(self._on_interim(transcript))

        except Exception as e:
            logger.error(f"Error processing transcript: {e}")

    async def _on_utterance_end(self, _client, utterance_end, **kwargs):
        if self._current_text:
            logger.info(f"Utterance ended: '{self._current_text}' [{self._detected_language}]")
            result = TranscriptResult(
                text=self._current_text,
                language=self._detected_language,
                is_final=True,
                confidence=1.0,
            )
            asyncio.create_task(self._on_final(result))
            self._current_text = ""

    async def _on_error(self, _client, error, **kwargs):
        logger.error(f"Deepgram error: {error}")

    async def __aexit__(self, *args):
        if self._connection:
            await self._connection.finish()
            logger.info("Deepgram STT session closed")