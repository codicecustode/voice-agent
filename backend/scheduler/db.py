"""
scheduler/db.py — Database models and async engine setup.

Tables:
  patients      — patient records + language preference
  doctors       — doctor profiles
  slots         — available time slots per doctor
  appointments  — booked appointments
  visit_summaries — LLM-generated summaries of past visits (for long-term memory)
  campaigns     — outbound call campaigns
  campaign_contacts — individual patients in a campaign
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    String, Text, text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from config import settings


# ── Engine ────────────────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    echo=False,  # set True to see SQL queries during debugging
)

async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncSession:
    """Dependency-injectable session for FastAPI routes."""
    async with async_session_factory() as session:
        yield session


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Models ────────────────────────────────────────────────────────────────────

class Patient(Base):
    __tablename__ = "patients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    phone = Column(String(20), unique=True, nullable=False)
    language = Column(String(10), default="en")   # "en" | "hi" | "ta"
    created_at = Column(DateTime, default=datetime.utcnow)

    appointments = relationship("Appointment", back_populates="patient")
    visit_summaries = relationship("VisitSummary", back_populates="patient")


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    specialty = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)

    slots = relationship("Slot", back_populates="doctor")


class Slot(Base):
    __tablename__ = "slots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    is_booked = Column(Boolean, default=False)

    doctor = relationship("Doctor", back_populates="slots")
    appointment = relationship("Appointment", back_populates="slot", uselist=False)


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    slot_id = Column(UUID(as_uuid=True), ForeignKey("slots.id"), nullable=False)
    reason = Column(Text)
    status = Column(String(20), default="confirmed")  # confirmed | cancelled | rescheduled
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="appointments")
    slot = relationship("Slot", back_populates="appointment")


class VisitSummary(Base):
    """LLM-generated summary stored after each completed call — feeds long-term memory."""
    __tablename__ = "visit_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="visit_summaries")


class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    campaign_type = Column(String(50), nullable=False)   # "reminder" | "followup" | "checkup"
    message_template = Column(Text, nullable=False)
    scheduled_at = Column(DateTime, nullable=False)
    status = Column(String(20), default="pending")       # pending | running | done
    created_at = Column(DateTime, default=datetime.utcnow)

    contacts = relationship("CampaignContact", back_populates="campaign")


class CampaignContact(Base):
    __tablename__ = "campaign_contacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(UUID(as_uuid=True), ForeignKey("campaigns.id"), nullable=False)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    status = Column(String(20), default="pending")       # pending | called | booked | declined | failed
    called_at = Column(DateTime)
    outcome_notes = Column(Text)

    campaign = relationship("Campaign", back_populates="contacts")


# ── DB init ───────────────────────────────────────────────────────────────────

async def init_db():
    """Create all tables and seed demo data."""
    async with engine.begin() as conn:
        # Enable pgvector extension (for future embedding search)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    await _seed_demo_data()


async def _seed_demo_data():
    """Insert a few doctors and slots so the agent has something to work with."""
    from datetime import timedelta

    async with async_session_factory() as session:
        # Check if already seeded
        result = await session.execute(text("SELECT COUNT(*) FROM doctors"))
        count = result.scalar()
        if count > 0:
            return

        # Create doctors
        doctors = [
            Doctor(name="Dr. Priya Sharma", specialty="General Physician"),
            Doctor(name="Dr. Ramesh Kumar", specialty="Cardiologist"),
            Doctor(name="Dr. Anita Nair", specialty="Dermatologist"),
        ]
        session.add_all(doctors)
        await session.flush()

        # Create slots: next 7 days, 9am–5pm, hourly
        slots = []
        base = datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0)
        for day_offset in range(1, 8):
            day = base + timedelta(days=day_offset)
            for hour in range(9, 17):
                for doctor in doctors:
                    start = day.replace(hour=hour)
                    slots.append(Slot(
                        doctor_id=doctor.id,
                        start_time=start,
                        end_time=start + timedelta(hours=1),
                    ))
        session.add_all(slots)

        # Demo patient
        session.add(Patient(
            name="Raj Patel",
            phone="+919876543210",
            language="hi",
        ))

        await session.commit()
