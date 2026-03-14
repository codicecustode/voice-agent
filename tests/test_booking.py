"""
tests/test_booking.py — Tests for appointment booking and conflict logic.

Run: pytest tests/test_booking.py -v
Requires: docker-compose up (Postgres + Redis)
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from uuid import uuid4

import pytest_asyncio
from sqlalchemy import text

from scheduler.db import init_db, async_session_factory, Doctor, Patient, Slot
from scheduler.booking import (
    book_appointment,
    cancel_appointment,
    get_available_slots,
    reschedule_appointment,
)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_db():
    await init_db()


@pytest_asyncio.fixture
async def test_data():
    """Create a doctor, patient, and two slots for testing."""
    async with async_session_factory() as session:
        doctor = Doctor(name="Dr. Test", specialty="General")
        patient = Patient(name="Test Patient", phone=f"+91{uuid4().int % 10**10:010d}")
        session.add_all([doctor, patient])
        await session.flush()

        future = datetime.utcnow() + timedelta(days=1)
        slot1 = Slot(doctor_id=doctor.id, start_time=future, end_time=future + timedelta(hours=1))
        slot2 = Slot(
            doctor_id=doctor.id,
            start_time=future + timedelta(hours=2),
            end_time=future + timedelta(hours=3),
        )
        session.add_all([slot1, slot2])
        await session.commit()

        return {
            "doctor_id": str(doctor.id),
            "patient_id": str(patient.id),
            "slot1_id": str(slot1.id),
            "slot2_id": str(slot2.id),
        }


@pytest.mark.asyncio
async def test_book_slot_success(test_data):
    result = await book_appointment(
        patient_id=test_data["patient_id"],
        slot_id=test_data["slot1_id"],
        reason="Checkup",
    )
    assert result["success"] is True
    assert "appointment_id" in result
    assert "Dr. Test" in result["message"]


@pytest.mark.asyncio
async def test_double_booking_conflict(test_data):
    """Booking an already-booked slot should return conflict + alternatives."""
    # slot1 was booked in the previous test — try to book it again
    result = await book_appointment(
        patient_id=test_data["patient_id"],
        slot_id=test_data["slot1_id"],
    )
    assert result["success"] is False
    assert result.get("conflict") is True
    assert "alternatives" in result


@pytest.mark.asyncio
async def test_cancel_appointment(test_data):
    # Book slot2 first
    book_result = await book_appointment(
        patient_id=test_data["patient_id"],
        slot_id=test_data["slot2_id"],
    )
    assert book_result["success"] is True

    # Now cancel it
    cancel_result = await cancel_appointment(book_result["appointment_id"])
    assert cancel_result["success"] is True

    # Slot should be free again — book it once more
    rebook_result = await book_appointment(
        patient_id=test_data["patient_id"],
        slot_id=test_data["slot2_id"],
    )
    assert rebook_result["success"] is True


@pytest.mark.asyncio
async def test_get_available_slots_by_specialty(test_data):
    result = await get_available_slots(specialty="General")
    assert result["success"] is True
    # At least some slots should be available
    assert isinstance(result["slots"], list)


@pytest.mark.asyncio
async def test_past_slot_rejected():
    """Booking a slot in the past should be rejected."""
    async with async_session_factory() as session:
        doctor = Doctor(name="Dr. Past", specialty="General")
        patient = Patient(name="Past Patient", phone=f"+91{uuid4().int % 10**10:010d}")
        session.add_all([doctor, patient])
        await session.flush()

        past_time = datetime.utcnow() - timedelta(hours=2)
        past_slot = Slot(
            doctor_id=doctor.id,
            start_time=past_time,
            end_time=past_time + timedelta(hours=1),
        )
        session.add(past_slot)
        await session.commit()

    result = await book_appointment(
        patient_id=str(patient.id),
        slot_id=str(past_slot.id),
    )
    assert result["success"] is False
    assert "past" in result["message"].lower()
