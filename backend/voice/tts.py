"""
voice/tts.py — Text-to-speech using ElevenLabs streaming API.

The key for low latency is STREAMING: we don't wait for ElevenLabs
to produce the full audio file. Instead, we request audio chunks
as the text comes in, and send the FIRST chunk back to the client
as fast as possible.

Latency strategy:
  1. Agent produces first sentence/clause → we flush it to TTS immediately.
  2. ElevenLabs streams back audio bytes chunk by chunk.
  3. We forward those bytes over WebSocket to the browser.
  4. Browser starts playing audio BEFORE the agent finishes generating text.

This "parallel pipeline" is what gets us under 450ms.
"""

import asyncio
from typing import AsyncIterator

import httpx
from loguru import logger

from config import settings
from agent.lang_detect import get_voice_id

# ElevenLabs streaming endpoint
ELEVENLABS_STREAM_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"

# Optimized TTS settings for low latency
TTS_SETTINGS = {
    "model_id": "eleven_turbo_v2_5",    # Fastest ElevenLabs model
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": False,      # Disable for speed
    },
    "output_format": "pcm_16000",        # Raw PCM, no MP3 decode overhead
}


class TTSClient:
    """Streaming TTS client."""

    def __init__(self):
        self._http = httpx.AsyncClient(timeout=30)

    async def stream_audio(
        self,
        text: str,
        language: str,
    ) -> AsyncIterator[bytes]:
        """
        Yields raw PCM audio chunks for the given text.
        Language determines which voice to use.

        Usage:
            async for chunk in tts.stream_audio("Hello", "en"):
                await websocket.send_bytes(chunk)
        """
        voice_id = get_voice_id(language)
        url = ELEVENLABS_STREAM_URL.format(voice_id=voice_id)

        headers = {
            "xi-api-key": settings.elevenlabs_api_key,
            "Content-Type": "application/json",
        }

        body = {
            "text": text,
            **TTS_SETTINGS,
        }

        logger.debug(f"TTS request [{language}] voice={voice_id}: '{text[:60]}...'")

        try:
            async with self._http.stream("POST", url, json=body, headers=headers) as response:
                if response.status_code != 200:
                    error = await response.aread()
                    logger.error(f"ElevenLabs error {response.status_code}: {error}")
                    return

                async for chunk in response.aiter_bytes(chunk_size=4096):
                    if chunk:
                        yield chunk

        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
            raise

    async def close(self):
        await self._http.aclose()


class SentenceBuffer:
    """
    Accumulates LLM text tokens and flushes complete sentences to TTS.

    This is the "parallel pipeline" trick:
    We don't wait for the full LLM response. As soon as we have a
    complete sentence (ending in . ! ? ।), we flush it to TTS.

    ।  is the Hindi/Tamil sentence-ending character (Devanagari Danda).
    """

    FLUSH_CHARS = {".", "!", "?", "।", "\n"}
    MIN_FLUSH_LENGTH = 20   # Don't flush very short fragments

    def __init__(self):
        self._buffer = ""

    def add(self, token: str) -> str | None:
        """
        Add a token. Returns a sentence to flush if ready, else None.
        """
        self._buffer += token

        # Check if we have a complete sentence
        for char in self.FLUSH_CHARS:
            if char in self._buffer and len(self._buffer) >= self.MIN_FLUSH_LENGTH:
                # Split at the first sentence boundary
                idx = self._buffer.find(char)
                sentence = self._buffer[:idx + 1].strip()
                self._buffer = self._buffer[idx + 1:]
                if sentence:
                    return sentence

        return None

    def flush_remaining(self) -> str | None:
        """Call this when the LLM stream finishes to flush any remaining text."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None
