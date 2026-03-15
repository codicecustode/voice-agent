"""voice/tts.py — Text-to-speech using Azure Cognitive Services Speech SDK."""

import asyncio
from xml.sax.saxutils import escape
from typing import AsyncIterator

import azure.cognitiveservices.speech as speechsdk
from loguru import logger

from config import settings, get_azure_tts_key, get_azure_tts_region
from agent.lang_detect import get_voice_id


class TTSClient:
    """Streaming TTS client."""

    def __init__(self):
        self._config_error_reported = False

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
        azure_key = get_azure_tts_key()
        azure_region = get_azure_tts_region()

        if not azure_key or not azure_region:
            if not self._config_error_reported:
                logger.error(
                    "Azure TTS is not configured. Set AZURE_TTS_KEY/AZURE_TTS_REGION "
                    "(or AZURE_SPEECH_KEY/AZURE_SPEECH_REGION)."
                )
                self._config_error_reported = True
            return

        voice_name = get_voice_id(language)

        ssml = (
            "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
            "xml:lang='en-US'>"
            f"<voice name='{voice_name}'>{escape(text)}</voice>"
            "</speak>"
        )

        logger.debug(f"TTS request [{language}] voice={voice_name}: '{text[:60]}...'")

        try:
            def synthesize_once() -> speechsdk.SpeechSynthesisResult:
                speech_config = speechsdk.SpeechConfig(
                    subscription=azure_key,
                    region=azure_region,
                )
                speech_config.speech_synthesis_voice_name = voice_name
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
                )
                synthesizer = speechsdk.SpeechSynthesizer(
                    speech_config=speech_config,
                    audio_config=None,
                )
                return synthesizer.speak_ssml_async(ssml).get()

            result = await asyncio.to_thread(synthesize_once)

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio = bytes(result.audio_data or b"")
                for idx in range(0, len(audio), 4096):
                    chunk = audio[idx : idx + 4096]
                    if chunk:
                        yield chunk
                return

            if result.reason == speechsdk.ResultReason.Canceled:
                details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
                logger.error(
                    f"Azure TTS canceled: reason={details.reason} code={details.error_code} details={details.error_details}"
                )
                return

            logger.error(f"Azure TTS failed with reason: {result.reason}")

        except Exception as e:
            logger.error(f"Azure TTS SDK error: {e}")
            raise

    async def close(self):
        return


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
