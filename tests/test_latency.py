"""
tests/test_latency.py — Tests for latency tracking.

Run: pytest tests/test_latency.py -v
"""

import asyncio
import time
import pytest

from monitoring.latency import LatencyTracker


def test_e2e_latency_under_target():
    """Simulate a fast pipeline — should be under 450ms."""
    tracker = LatencyTracker(session_id="test-1")
    tracker.mark_speech_end()
    time.sleep(0.05)   # 50ms STT
    tracker.mark_stt_final()
    time.sleep(0.15)   # 150ms LLM first token
    tracker.mark_llm_first_token()
    time.sleep(0.07)   # 70ms TTS first chunk
    e2e = tracker.mark_first_audio()
    assert e2e is not None
    assert e2e < 450, f"E2E latency {e2e:.1f}ms exceeded 450ms target"


def test_latency_stages_summed():
    """STT + LLM + TTS should equal roughly total E2E."""
    tracker = LatencyTracker(session_id="test-2")
    tracker.mark_speech_end()
    time.sleep(0.08)
    tracker.mark_stt_final()
    time.sleep(0.12)
    tracker.mark_llm_first_token()
    time.sleep(0.06)
    e2e = tracker.mark_first_audio()

    stt = tracker._stt_ms
    llm = tracker._llm_ms
    tts = tracker._tts_ms

    assert stt > 0 and llm > 0 and tts > 0
    # Sum of stages should roughly equal E2E (within 5ms rounding)
    assert abs((stt + llm + tts) - e2e) < 5
