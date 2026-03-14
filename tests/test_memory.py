"""
tests/test_memory.py — Tests for session and long-term memory.

Run: pytest tests/test_memory.py -v
"""

import asyncio
import pytest
from uuid import uuid4

from agent.memory import MemoryManager


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def memory():
    return MemoryManager()


@pytest.mark.asyncio
async def test_session_defaults(memory):
    """New session should have sensible defaults."""
    session_id = str(uuid4())
    state = await memory.get_session(session_id)
    assert state["language"] == "en"
    assert state["intent"] is None
    assert state["turn_history"] == []


@pytest.mark.asyncio
async def test_session_persist_and_load(memory):
    session_id = str(uuid4())
    await memory.update_session(session_id, language="hi", intent="book")
    loaded = await memory.get_session(session_id)
    assert loaded["language"] == "hi"
    assert loaded["intent"] == "book"


@pytest.mark.asyncio
async def test_turn_history_capped(memory):
    """Turn history should not exceed 10 entries."""
    session_id = str(uuid4())
    for i in range(15):
        await memory.append_turn(session_id, "user", f"message {i}")
    state = await memory.get_session(session_id)
    assert len(state["turn_history"]) == 10


@pytest.mark.asyncio
async def test_session_delete(memory):
    session_id = str(uuid4())
    await memory.update_session(session_id, language="ta")
    await memory.delete_session(session_id)
    state = await memory.get_session(session_id)
    # After deletion, should return defaults
    assert state["language"] == "en"


@pytest.mark.asyncio
async def test_patient_context_new(memory):
    """Unknown patient_id should return graceful 'no history' message."""
    ctx = await memory.get_patient_context(str(uuid4()))
    assert "no history" in ctx.lower() or "new patient" in ctx.lower()
