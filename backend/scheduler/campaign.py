"""
scheduler/campaign.py — Outbound call campaigns using Celery + Twilio.

How it works:
  1. A Campaign is created in the DB with a list of patients and a scheduled time.
  2. Celery Beat (a scheduler) checks every minute for due campaigns.
  3. For each patient in the campaign, we enqueue a Celery task.
  4. The task uses Twilio to place an outbound call.
  5. Twilio calls the patient, then connects to our voice agent WebSocket.
  6. The agent handles the conversation naturally (book/reschedule/decline).

To run:
  celery -A scheduler.campaign worker --loglevel=info
  celery -A scheduler.campaign beat --loglevel=info
"""

import asyncio
import json
import os
from datetime import datetime, timedelta

from celery import Celery
from loguru import logger
from twilio.rest import Client as TwilioClient

from config import settings

# ── Celery app ────────────────────────────────────────────────────────────────

celery_app = Celery(
    "voice_agent",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
    # Beat schedule: check for due campaigns every 60 seconds
    beat_schedule={
        "dispatch-due-campaigns": {
            "task": "scheduler.campaign.dispatch_due_campaigns",
            "schedule": 60.0,  # seconds
        }
    },
)


# ── Twilio helper ─────────────────────────────────────────────────────────────

def get_twilio_client() -> TwilioClient:
    return TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)


def build_twiml_for_outbound(campaign_type: str, patient_name: str, appointment_info: str) -> str:
    """
    TwiML that Twilio executes when the patient picks up.
    We use <Connect><Stream> to bridge the call audio to our WebSocket agent.
    """
    ws_url = f"{settings.public_base_url.replace('https', 'wss')}/ws/outbound"

    # We pass campaign context as custom parameters to the WebSocket
    context = json.dumps({
        "campaign_type": campaign_type,
        "patient_name": patient_name,
        "appointment_info": appointment_info,
    })

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="campaign_context" value='{context}'/>
        </Stream>
    </Connect>
</Response>"""


# ── Celery tasks ──────────────────────────────────────────────────────────────

@celery_app.task(name="scheduler.campaign.dispatch_due_campaigns", bind=True)
def dispatch_due_campaigns(self):
    """
    Periodic task: find all pending campaigns that are due and
    enqueue individual call tasks for each patient contact.
    """
    async def _run():
        from scheduler.db import async_session_factory, Campaign, CampaignContact
        from sqlalchemy import select, and_

        async with async_session_factory() as session:
            now = datetime.utcnow()
            result = await session.execute(
                select(Campaign).where(
                    and_(
                        Campaign.status == "pending",
                        Campaign.scheduled_at <= now,
                    )
                )
            )
            due_campaigns = result.scalars().all()

            for campaign in due_campaigns:
                logger.info(f"Dispatching campaign: {campaign.name} ({campaign.id})")
                campaign.status = "running"

                # Enqueue a call task for each contact
                contacts_result = await session.execute(
                    select(CampaignContact).where(
                        CampaignContact.campaign_id == campaign.id,
                        CampaignContact.status == "pending",
                    )
                )
                contacts = contacts_result.scalars().all()

                for contact in contacts:
                    place_outbound_call.apply_async(
                        args=[str(campaign.id), str(contact.id)],
                        countdown=0,    # start immediately
                    )

            await session.commit()

    asyncio.get_event_loop().run_until_complete(_run())


@celery_app.task(
    name="scheduler.campaign.place_outbound_call",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def place_outbound_call(self, campaign_id: str, contact_id: str):
    """
    Place a single outbound call to a patient for a campaign.

    Retries up to 3 times if the call fails (e.g. network error).
    """
    async def _run():
        from scheduler.db import async_session_factory, Campaign, CampaignContact, Patient, Appointment, Slot, Doctor
        from sqlalchemy import select, and_

        async with async_session_factory() as session:
            # Load campaign + contact + patient
            contact = await session.get(CampaignContact, contact_id)
            campaign = await session.get(Campaign, campaign_id)
            patient = await session.get(Patient, contact.patient_id)

            if not patient or not patient.phone:
                logger.error(f"No phone for patient {contact.patient_id}")
                contact.status = "failed"
                await session.commit()
                return

            # Build appointment info for the campaign context
            appointment_info = campaign.message_template.format(
                patient_name=patient.name,
                # Additional fields can be added based on campaign type
            )

            logger.info(f"Calling {patient.phone} for campaign '{campaign.name}'")

            try:
                twilio = get_twilio_client()
                twiml = build_twiml_for_outbound(
                    campaign.campaign_type,
                    patient.name,
                    appointment_info,
                )

                call = twilio.calls.create(
                    to=patient.phone,
                    from_=settings.twilio_from_number,
                    twiml=twiml,
                    timeout=30,
                    status_callback=f"{settings.public_base_url}/webhooks/twilio/status",
                    status_callback_method="POST",
                )

                contact.status = "called"
                contact.called_at = datetime.utcnow()
                contact.outcome_notes = f"Twilio SID: {call.sid}"
                await session.commit()
                logger.info(f"Call placed: {call.sid} to {patient.phone}")

            except Exception as exc:
                logger.error(f"Failed to call {patient.phone}: {exc}")
                contact.status = "failed"
                await session.commit()
                raise self.retry(exc=exc)

    asyncio.get_event_loop().run_until_complete(_run())


@celery_app.task(name="scheduler.campaign.update_contact_outcome")
def update_contact_outcome(contact_id: str, outcome: str, notes: str = ""):
    """
    Called by the WebSocket handler after the outbound conversation ends.
    Records whether the patient booked, rescheduled, or declined.

    outcome: "booked" | "rescheduled" | "declined" | "no_answer" | "failed"
    """
    async def _run():
        from scheduler.db import async_session_factory, CampaignContact

        async with async_session_factory() as session:
            contact = await session.get(CampaignContact, contact_id)
            if contact:
                contact.status = outcome
                contact.outcome_notes = notes
                await session.commit()
                logger.info(f"Contact {contact_id} outcome: {outcome}")

    asyncio.get_event_loop().run_until_complete(_run())
