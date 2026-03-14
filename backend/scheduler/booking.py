"""
scheduler/booking.py — All appointment operations.

Key design decisions:
  1. Row-level locking (SELECT FOR UPDATE) prevents race conditions
     when two calls try to book the same slot simultaneously.
  2. All functions return structured dicts that the agent's tools.py
     passes back to the LLM as tool results.
  3. Conflict resolution always offers alternatives so the LLM
     can suggest them to the patient naturally.
"""

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from loguru import logger
from sqlalchemy import select, and_, text
from sqlalchemy.ext.asyncio import AsyncSession

from scheduler.db import Appointment, Doctor, Patient, Slot, async_session_factory


# ── Helper ────────────────────────────────────────────────────────────────────

def _format_slot(slot: Slot, doctor: Doctor) -> dict:
    """Format a slot for LLM consumption."""
    return {
        "slot_id": str(slot.id),
        "doctor_id": str(doctor.id),
        "doctor_name": doctor.name,
        "specialty": doctor.specialty,
        "start_time": slot.start_time.isoformat(),
        "end_time": slot.end_time.isoformat(),
        "display": f"{slot.start_time.strftime('%A, %d %B %Y at %I:%M %p')} with {doctor.name}",
    }


async def _find_alternatives(
    session: AsyncSession,
    doctor_id: str,
    near_time: datetime,
    limit: int = 3,
) -> list[dict]:
    """Find available slots near a requested time when the exact slot is taken."""
    result = await session.execute(
        select(Slot, Doctor)
        .join(Doctor, Slot.doctor_id == Doctor.id)
        .where(
            and_(
                Slot.doctor_id == doctor_id,
                Slot.is_booked == False,
                Slot.start_time > datetime.utcnow(),
                Slot.start_time >= near_time - timedelta(days=3),
                Slot.start_time <= near_time + timedelta(days=3),
            )
        )
        .order_by(Slot.start_time)
        .limit(limit)
    )
    return [_format_slot(s, d) for s, d in result.all()]


# ── Public booking functions ──────────────────────────────────────────────────

async def get_available_slots(
    doctor_name: str | None = None,
    specialty: str | None = None,
    date_hint: str | None = None,
    limit: int = 5,
) -> dict:
    """
    Find available slots, optionally filtered by doctor name or specialty.
    date_hint is a natural language string like "tomorrow" or "next Monday"
    — the LLM should parse this to an ISO date before calling the tool.
    """
    async with async_session_factory() as session:
        query = (
            select(Slot, Doctor)
            .join(Doctor, Slot.doctor_id == Doctor.id)
            .where(
                and_(
                    Slot.is_booked == False,
                    Slot.start_time > datetime.utcnow(),
                    Doctor.is_active == True,
                )
            )
        )

        if doctor_name:
            query = query.where(Doctor.name.ilike(f"%{doctor_name}%"))
        if specialty:
            query = query.where(Doctor.specialty.ilike(f"%{specialty}%"))
        if date_hint:
            try:
                target_date = datetime.fromisoformat(date_hint)
                query = query.where(
                    and_(
                        Slot.start_time >= target_date.replace(hour=0, minute=0),
                        Slot.start_time < target_date.replace(hour=23, minute=59),
                    )
                )
            except ValueError:
                pass  # LLM sent bad date — just ignore the filter

        query = query.order_by(Slot.start_time).limit(limit)
        results = await session.execute(query)
        slots = [_format_slot(s, d) for s, d in results.all()]

        if not slots:
            return {
                "success": False,
                "message": "No available slots found for the given criteria.",
                "slots": [],
            }

        return {"success": True, "slots": slots}


async def book_appointment(
    patient_id: str,
    slot_id: str,
    reason: str = "",
) -> dict:
    """
    Book a specific slot. Uses SELECT FOR UPDATE to prevent double-booking.

    Returns:
      success=True  → appointment details
      success=False, conflict=True → conflict message + alternative slots
    """
    async with async_session_factory() as session:
        async with session.begin():
            # Lock this specific slot row for the duration of the transaction
            result = await session.execute(
                select(Slot)
                .where(Slot.id == slot_id)
                .with_for_update()  # ← row-level lock: prevents race conditions
            )
            slot = result.scalar_one_or_none()

            if not slot:
                return {"success": False, "message": "Slot not found."}

            # Past time check
            if slot.start_time <= datetime.utcnow():
                return {
                    "success": False,
                    "message": "That time slot is in the past. Please choose a future slot.",
                }

            # Already booked
            if slot.is_booked:
                alternatives = await _find_alternatives(
                    session, str(slot.doctor_id), slot.start_time
                )
                return {
                    "success": False,
                    "conflict": True,
                    "message": "This slot was just booked by someone else.",
                    "alternatives": alternatives,
                }

            # Mark slot as booked
            slot.is_booked = True

            # Create appointment record
            appointment = Appointment(
                patient_id=patient_id,
                slot_id=slot_id,
                reason=reason,
                status="confirmed",
            )
            session.add(appointment)
            await session.flush()

            # Load doctor info for the response
            doctor = await session.get(Doctor, slot.doctor_id)

            logger.info(f"Booked appointment {appointment.id} for patient {patient_id}")
            return {
                "success": True,
                "appointment_id": str(appointment.id),
                "slot": _format_slot(slot, doctor),
                "reason": reason,
                "message": f"Appointment confirmed with {doctor.name} on {slot.start_time.strftime('%A, %d %B at %I:%M %p')}.",
            }


async def reschedule_appointment(
    appointment_id: str,
    new_slot_id: str,
) -> dict:
    """Cancel the old slot and book the new one atomically."""
    async with async_session_factory() as session:
        async with session.begin():
            # Load old appointment
            old_appt = await session.get(Appointment, appointment_id)
            if not old_appt:
                return {"success": False, "message": "Appointment not found."}
            if old_appt.status == "cancelled":
                return {"success": False, "message": "This appointment is already cancelled."}

            # Free the old slot
            old_slot = await session.get(Slot, old_appt.slot_id)
            if old_slot:
                old_slot.is_booked = False

            # Lock and book new slot
            result = await session.execute(
                select(Slot).where(Slot.id == new_slot_id).with_for_update()
            )
            new_slot = result.scalar_one_or_none()
            if not new_slot or new_slot.is_booked:
                # Rollback by re-booking old slot
                if old_slot:
                    old_slot.is_booked = True
                return {
                    "success": False,
                    "conflict": True,
                    "message": "The requested new slot is no longer available.",
                }

            new_slot.is_booked = True
            old_appt.slot_id = new_slot_id
            old_appt.status = "rescheduled"

            doctor = await session.get(Doctor, new_slot.doctor_id)
            logger.info(f"Rescheduled appointment {appointment_id} to slot {new_slot_id}")
            return {
                "success": True,
                "appointment_id": appointment_id,
                "new_slot": _format_slot(new_slot, doctor),
                "message": f"Appointment rescheduled to {new_slot.start_time.strftime('%A, %d %B at %I:%M %p')} with {doctor.name}.",
            }


async def cancel_appointment(appointment_id: str) -> dict:
    """Cancel an appointment and free the slot."""
    async with async_session_factory() as session:
        async with session.begin():
            appt = await session.get(Appointment, appointment_id)
            if not appt:
                return {"success": False, "message": "Appointment not found."}
            if appt.status == "cancelled":
                return {"success": False, "message": "This appointment is already cancelled."}

            # Free the slot
            slot = await session.get(Slot, appt.slot_id)
            if slot:
                slot.is_booked = False

            appt.status = "cancelled"
            logger.info(f"Cancelled appointment {appointment_id}")
            return {
                "success": True,
                "appointment_id": appointment_id,
                "message": "Appointment has been cancelled successfully.",
            }


async def get_patient_appointments(patient_id: str) -> dict:
    """Fetch upcoming appointments for a patient."""
    async with async_session_factory() as session:
        result = await session.execute(
            select(Appointment, Slot, Doctor)
            .join(Slot, Appointment.slot_id == Slot.id)
            .join(Doctor, Slot.doctor_id == Doctor.id)
            .where(
                and_(
                    Appointment.patient_id == patient_id,
                    Appointment.status.in_(["confirmed", "rescheduled"]),
                    Slot.start_time > datetime.utcnow(),
                )
            )
            .order_by(Slot.start_time)
        )
        appointments = result.all()

        if not appointments:
            return {"success": True, "appointments": [], "message": "No upcoming appointments."}

        return {
            "success": True,
            "appointments": [
                {
                    "appointment_id": str(appt.id),
                    "slot": _format_slot(slot, doctor),
                    "reason": appt.reason,
                    "status": appt.status,
                }
                for appt, slot, doctor in appointments
            ],
        }
