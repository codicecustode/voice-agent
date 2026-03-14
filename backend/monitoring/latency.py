"""
monitoring/latency.py — Measure and report end-to-end response latency.

The assignment requires: "latency must be measured, logged, and discussed."

We track these milestones per request:
  t0: speech_end    — Deepgram fires UtteranceEnd
  t1: stt_final     — we receive the final transcript text
  t2: llm_first_tok — first text token from the LLM
  t3: first_audio   — first audio bytes sent to the client

Target: t3 - t0 < 450ms
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from prometheus_client import Histogram, Counter, start_http_server


# ── Prometheus metrics ────────────────────────────────────────────────────────

E2E_LATENCY = Histogram(
    "voice_agent_e2e_latency_ms",
    "Total latency from speech end to first audio byte (ms)",
    buckets=[100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 750, 1000, 2000],
)

STT_LATENCY = Histogram(
    "voice_agent_stt_latency_ms",
    "Deepgram STT time: speech end to final transcript (ms)",
    buckets=[20, 40, 60, 80, 100, 150, 200],
)

LLM_LATENCY = Histogram(
    "voice_agent_llm_latency_ms",
    "LLM time: transcript received to first token (ms)",
    buckets=[50, 100, 150, 200, 250, 300, 400, 500],
)

TTS_LATENCY = Histogram(
    "voice_agent_tts_latency_ms",
    "TTS time: text sent to first audio chunk (ms)",
    buckets=[30, 50, 80, 100, 150, 200, 300],
)

TOOL_CALL_COUNTER = Counter(
    "voice_agent_tool_calls_total",
    "Number of tool calls made",
    labelnames=["tool_name", "success"],
)

BOOKING_COUNTER = Counter(
    "voice_agent_bookings_total",
    "Number of successful bookings",
)

CONFLICT_COUNTER = Counter(
    "voice_agent_booking_conflicts_total",
    "Number of booking conflicts encountered",
)


def start_metrics_server(port: int = 8001):
    """Start the Prometheus metrics HTTP server on a separate port."""
    try:
        start_http_server(port)
        logger.info(f"Prometheus metrics available at http://localhost:{port}/metrics")
    except Exception as e:
        logger.warning(f"Could not start metrics server: {e}")


# ── Per-request tracker ───────────────────────────────────────────────────────

@dataclass
class LatencyTracker:
    """
    Tracks timing milestones for a single request.
    Create one per WebSocket message and pass it through the pipeline.
    """
    session_id: str = ""
    _t_speech_end: float = field(default_factory=time.monotonic)
    _t_stt_final: Optional[float] = None
    _t_llm_first: Optional[float] = None
    _t_first_audio: Optional[float] = None

    def mark_speech_end(self):
        """Call when Deepgram fires UtteranceEnd."""
        self._t_speech_end = time.monotonic()

    def mark_stt_final(self):
        """Call when we have the final transcript text."""
        self._t_stt_final = time.monotonic()
        if self._t_speech_end:
            ms = (self._t_stt_final - self._t_speech_end) * 1000
            STT_LATENCY.observe(ms)
            logger.debug(f"[{self.session_id}] STT latency: {ms:.1f}ms")

    def mark_llm_first_token(self):
        """Call when the LLM emits its first text token."""
        self._t_llm_first = time.monotonic()
        if self._t_stt_final:
            ms = (self._t_llm_first - self._t_stt_final) * 1000
            LLM_LATENCY.observe(ms)
            logger.debug(f"[{self.session_id}] LLM first-token latency: {ms:.1f}ms")

    def mark_first_audio(self):
        """
        Call when the FIRST audio bytes are sent to the client.
        This is the final E2E measurement point.
        """
        self._t_first_audio = time.monotonic()

        if self._t_llm_first:
            tts_ms = (self._t_first_audio - self._t_llm_first) * 1000
            TTS_LATENCY.observe(tts_ms)
            logger.debug(f"[{self.session_id}] TTS first-chunk latency: {tts_ms:.1f}ms")

        if self._t_speech_end:
            e2e_ms = (self._t_first_audio - self._t_speech_end) * 1000
            E2E_LATENCY.observe(e2e_ms)

            status = "✅ UNDER TARGET" if e2e_ms < 450 else "⚠️ OVER TARGET"
            logger.info(
                f"[{self.session_id}] E2E latency: {e2e_ms:.1f}ms {status} | "
                f"STT: {self._stt_ms:.0f}ms | "
                f"LLM: {self._llm_ms:.0f}ms | "
                f"TTS: {self._tts_ms:.0f}ms"
            )
            return e2e_ms

        return None

    @property
    def _stt_ms(self) -> float:
        if self._t_stt_final and self._t_speech_end:
            return (self._t_stt_final - self._t_speech_end) * 1000
        return 0.0

    @property
    def _llm_ms(self) -> float:
        if self._t_llm_first and self._t_stt_final:
            return (self._t_llm_first - self._t_stt_final) * 1000
        return 0.0

    @property
    def _tts_ms(self) -> float:
        if self._t_first_audio and self._t_llm_first:
            return (self._t_first_audio - self._t_llm_first) * 1000
        return 0.0
