"""
agent/memory.py — Two-tier contextual memory.

=== Tier 1: Session memory (Redis) ===
  - Scope: one conversation session
  - Storage: Redis key  session:{session_id}
  - TTL: 30 minutes of inactivity
  - Contains:
      - language: detected/preferred language this session
      - intent: what the patient is trying to do (book / reschedule / cancel)
      - slots: collected info (doctor name, date, reason, etc.)
      - pending_confirmation: a booking we're waiting for the patient to confirm
      - turn_history: last N messages for short-term context

=== Tier 2: Long-term memory (PostgreSQL) ===
  - Scope: across all sessions, permanently
  - Storage: Postgres  patients, visit_summaries tables
  - Contains:
      - patient profile (name, phone, language preference)
      - past visit summaries (LLM-generated after each call)
      - appointment history

Why this design?
  Redis is in-memory and blazing fast (<1ms) — perfect for
  the active conversation where every ms counts.
  Postgres is durable and queryable — perfect for patient history
  that needs to survive server restarts.
"""

import json
from datetime import datetime
from typing import Any

import redis.asyncio as aioredis
from loguru import logger
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from scheduler.db import Patient, VisitSummary, async_session_factory


# ── Session state schema ──────────────────────────────────────────────────────
# This is the structure stored in Redis per session.
DEFAULT_SESSION = {
    "language": "en",
    "intent": None,                  # "book" | "reschedule" | "cancel" | "query"
    "slots": {},                     # collected info: {doctor_id, slot_id, reason, ...}
    "pending_confirmation": None,    # booking details awaiting "yes/no"
    "turn_history": [],              # list of {role, content} — last 10 turns
    "patient_id": None,
    "created_at": None,
}


class MemoryManager:

    def __init__(self):
        self._redis = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

    # ── Session memory ────────────────────────────────────────────────────────

    async def get_session(self, session_id: str) -> dict:
        """Load session state from Redis. Returns defaults if not found."""
        key = f"session:{session_id}"
        raw = await self._redis.get(key)
        if raw:
            return json.loads(raw)
        # New session
        return {**DEFAULT_SESSION, "created_at": datetime.utcnow().isoformat()}

    async def save_session(self, session_id: str, state: dict) -> None:
        """Persist session state to Redis with TTL refresh."""
        key = f"session:{session_id}"
        await self._redis.setex(
            key,
            settings.session_ttl_seconds,
            json.dumps(state, default=str),
        )

    async def update_session(self, session_id: str, **updates) -> dict:
        """Load, update specific fields, save back. Returns updated state."""
        state = await self.get_session(session_id)
        state.update(updates)
        await self.save_session(session_id, state)
        return state

    async def append_turn(
        self, session_id: str, role: str, content: str, max_turns: int = 10
    ) -> None:
        """Add a conversation turn to session history, capping at max_turns."""
        state = await self.get_session(session_id)
        history = state.get("turn_history", [])
        history.append({"role": role, "content": content})
        # Keep only the most recent turns to control prompt size
        state["turn_history"] = history[-max_turns:]
        await self.save_session(session_id, state)

    async def delete_session(self, session_id: str) -> None:
        await self._redis.delete(f"session:{session_id}")

    # ── Long-term memory ──────────────────────────────────────────────────────

    async def get_patient_context(self, patient_id: str) -> str:
        """
        Build a context string injected into the system prompt.
        Includes the patient profile and their last 3 visit summaries.

        Returns a formatted string like:
          Patient: Raj Patel | Language: Hindi
          Past visits:
          - [2024-01-15] Booked cardiology follow-up. Patient mentioned chest pain.
          - [2024-01-01] Cancelled appointment due to travel. Requested rescheduling.
        """
        async with async_session_factory() as session:
            # Load patient profile
            patient = await session.get(Patient, patient_id)
            if not patient:
                return "New patient — no history on file."

            # Load last 3 visit summaries
            result = await session.execute(
                select(VisitSummary)
                .where(VisitSummary.patient_id == patient_id)
                .order_by(desc(VisitSummary.created_at))
                .limit(3)
            )
            summaries = result.scalars().all()

            lines = [
                f"Patient: {patient.name} | Phone: {patient.phone} | Language preference: {patient.language}",
            ]
            if summaries:
                lines.append("Past interactions:")
                for s in summaries:
                    date = s.created_at.strftime("%Y-%m-%d")
                    lines.append(f"  - [{date}] {s.summary}")
            else:
                lines.append("No past interactions on record.")

            return "\n".join(lines)

    async def save_visit_summary(self, patient_id: str, summary: str) -> None:
        """
        Call this at the end of each session with an LLM-generated summary.
        This is what builds the patient's long-term memory.
        """
        async with async_session_factory() as session:
            session.add(VisitSummary(patient_id=patient_id, summary=summary))
            await session.commit()
        logger.info(f"Saved visit summary for patient {patient_id}")

    async def update_patient_language(self, patient_id: str, language: str) -> None:
        """Persist language preference to the patient's profile."""
        async with async_session_factory() as session:
            patient = await session.get(Patient, patient_id)
            if patient and patient.language != language:
                patient.language = language
                await session.commit()
                logger.info(f"Updated language for patient {patient_id} to {language}")

    async def get_or_create_patient(self, phone: str, name: str = "Unknown") -> str:
        """Look up patient by phone. Creates a new record if not found. Returns patient_id."""
        async with async_session_factory() as session:
            result = await session.execute(
                select(Patient).where(Patient.phone == phone)
            )
            patient = result.scalar_one_or_none()
            if patient:
                return str(patient.id)
            # Create new patient
            new_patient = Patient(name=name, phone=phone, language="en")
            session.add(new_patient)
            await session.commit()
            logger.info(f"Created new patient record for {phone}")
            return str(new_patient.id)

    async def close(self):
        await self._redis.aclose()
