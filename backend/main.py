"""
main.py — FastAPI application and WebSocket endpoints.

Two WebSocket endpoints:
  /ws/inbound/{session_id}   — patient-initiated calls (browser/phone)
  /ws/outbound               — Twilio-bridged outbound campaign calls

The WebSocket message flow:
  Browser → sends raw PCM audio bytes
  Server  → forwards to Deepgram STT
  Deepgram → sends transcript event
  Server  → runs agent orchestrator
  Orchestrator → streams text tokens
    Server  → streams to Azure Speech TTS
    Azure Speech → streams audio chunks
  Server  → sends audio bytes back to browser

Barge-in: If Deepgram detects speech while we're still sending TTS audio,
we cancel the current TTS stream and start processing the new utterance.
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Optional
import sys

# Fix for Windows ProactorEventLoop conflict with websockets
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import make_asgi_app

from config import settings
from scheduler.db import init_db
from agent.orchestrator import AgentOrchestrator
from agent.memory import MemoryManager
from voice.stt import STTClient, TranscriptResult
from voice.tts import TTSClient, SentenceBuffer
from monitoring.latency import LatencyTracker, start_metrics_server


# ── Startup ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Voice Agent server...")
    await init_db()
    start_metrics_server(port=8001)
    logger.info("Database initialized. Server ready.")
    yield
    logger.info("Shutting down...")


app = FastAPI(title="Clinical Voice Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ── HTTP routes ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/patients")
async def create_patient(body: dict):
    """Create or look up a patient. Returns patient_id for session init."""
    memory = MemoryManager()
    patient_id = await memory.get_or_create_patient(
        phone=body["phone"],
        name=body.get("name", "Unknown"),
    )
    return {"patient_id": patient_id}


@app.post("/api/campaigns")
async def create_campaign(body: dict):
    """
    Create an outbound campaign.
    Body: {name, campaign_type, message_template, scheduled_at, patient_ids: []}
    """
    from scheduler.db import async_session_factory, Campaign, CampaignContact
    from datetime import datetime

    async with async_session_factory() as session:
        campaign = Campaign(
            name=body["name"],
            campaign_type=body["campaign_type"],
            message_template=body["message_template"],
            scheduled_at=datetime.fromisoformat(body["scheduled_at"]),
        )
        session.add(campaign)
        await session.flush()

        for patient_id in body.get("patient_ids", []):
            session.add(CampaignContact(
                campaign_id=campaign.id,
                patient_id=patient_id,
            ))
        await session.commit()
        return {"campaign_id": str(campaign.id), "status": "created"}


@app.post("/webhooks/twilio/status")
async def twilio_status_callback(body: dict):
    """Twilio calls this when a call status changes (completed, failed, etc.)."""
    logger.info(f"Twilio status: {body.get('CallStatus')} SID={body.get('CallSid')}")
    return {"ok": True}


# ── WebSocket connection handler ──────────────────────────────────────────────

class VoiceSession:
    """
    Manages a single real-time voice session.
    Handles the pipeline: audio → STT → agent → TTS → audio.
    """

    def __init__(
        self,
        websocket: WebSocket,
        session_id: str,
        patient_id: str,
        is_outbound: bool = False,
        campaign_context: Optional[dict] = None,
    ):
        self.ws = websocket
        self.session_id = session_id
        self.patient_id = patient_id
        self.is_outbound = is_outbound
        self.campaign_context = campaign_context

        self.stt = STTClient()
        self.tts = TTSClient()
        self.agent = AgentOrchestrator()
        self.memory = MemoryManager()
        self.latency = LatencyTracker(session_id=session_id)

        # Barge-in: track if TTS is currently playing
        self._tts_task: Optional[asyncio.Task] = None
        self._is_speaking = False  # True = agent is currently sending audio

    async def run(self):
        """Main session loop."""
        try:
            # For outbound calls, send a greeting first
            if self.is_outbound and self.campaign_context:
                await self._send_outbound_greeting()

            # Start STT streaming session
            async with self.stt.session(
                on_interim=self._on_interim_transcript,
                on_final=self._on_final_transcript,
            ) as send_audio:
                # Forward audio chunks from the WebSocket to Deepgram
                async for message in self.ws.iter_bytes():
                    await send_audio(message)

        except WebSocketDisconnect:
            logger.info(f"Session {self.session_id} disconnected")
        except Exception as e:
            logger.exception(f"Session {self.session_id} failed: {e}")
            try:
                await self.ws.send_text(json.dumps({
                    "type": "error",
                    "message": "Speech service unavailable",
                }))
            except Exception:
                pass
        finally:
            await self._cleanup()

    async def _on_interim_transcript(self, result: TranscriptResult):
        """
        Called with partial transcripts. Used for barge-in detection.
        If the agent is speaking and we detect meaningful speech,
        we cancel the current TTS output.
        """
        if self._is_speaking and len(result.text) > 5:
            logger.info(f"Barge-in detected: '{result.text}' — cancelling TTS")
            if self._tts_task and not self._tts_task.done():
                self._tts_task.cancel()
            self._is_speaking = False
            # Signal to client to stop audio playback
            await self.ws.send_text(json.dumps({"type": "barge_in"}))

    async def _on_final_transcript(self, result: TranscriptResult):
        """
        Called when Deepgram fires UtteranceEnd with the final transcript.
        This triggers the full agent pipeline.
        """
        self.latency.mark_stt_final()

        # Update language in session if detected
        await self.memory.update_session(
            self.session_id,
            language=result.language,
        )

        # Send acknowledgment to client (shows we heard them)
        await self.ws.send_text(json.dumps({
            "type": "transcript",
            "text": result.text,
            "language": result.language,
        }))

        # Run agent + TTS as a cancellable task (allows barge-in)
        self._tts_task = asyncio.create_task(
            self._process_and_speak(result.text, result.language)
        )

    async def _process_and_speak(self, transcript: str, language: str):
        """
        Run the agent, collect text, stream to TTS, send audio to client.

        PARALLEL PIPELINE:
        We don't wait for the full agent response before starting TTS.
        We accumulate tokens until we have a complete sentence, then
        flush that sentence to TTS immediately.
        """
        self._is_speaking = True
        sentence_buffer = SentenceBuffer()
        first_audio_sent = False

        try:
            async for token in self.agent.process(
                transcript,
                self.session_id,
                self.patient_id,
                self.latency,
            ):
                # Add token to buffer; get back a sentence when ready
                sentence = sentence_buffer.add(token)

                if sentence:
                    # Stream this sentence to TTS
                    async for audio_chunk in self.tts.stream_audio(sentence, language):
                        if not first_audio_sent:
                            self.latency.mark_first_audio()
                            first_audio_sent = True
                        await self.ws.send_bytes(audio_chunk)

            # Flush any remaining text
            remaining = sentence_buffer.flush_remaining()
            if remaining:
                async for audio_chunk in self.tts.stream_audio(remaining, language):
                    if not first_audio_sent:
                        self.latency.mark_first_audio()
                        first_audio_sent = True
                    await self.ws.send_bytes(audio_chunk)

            # Signal audio complete
            await self.ws.send_text(json.dumps({"type": "audio_end"}))

        except asyncio.CancelledError:
            # Barge-in cancelled this task — expected
            logger.debug(f"TTS task cancelled (barge-in) for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error in process_and_speak: {e}")
        finally:
            self._is_speaking = False

    async def _send_outbound_greeting(self):
        """For outbound calls, the agent speaks first."""
        ctx = self.campaign_context or {}
        patient_name = ctx.get("patient_name", "there")
        campaign_type = ctx.get("campaign_type", "reminder")
        appointment_info = ctx.get("appointment_info", "")

        if campaign_type == "reminder":
            greeting = (
                f"Hello, am I speaking with {patient_name}? "
                f"This is a call from your healthcare provider. "
                f"{appointment_info} "
                f"Would you like to confirm, reschedule, or cancel?"
            )
        else:
            greeting = (
                f"Hello {patient_name}, this is a follow-up call from your healthcare provider. "
                f"How have you been since your last visit? "
                f"Would you like to schedule an appointment?"
            )

        # Get language preference for this patient
        session = await self.memory.get_session(self.session_id)
        language = session.get("language", "en")

        async for audio_chunk in self.tts.stream_audio(greeting, language):
            await self.ws.send_bytes(audio_chunk)
        await self.ws.send_text(json.dumps({"type": "audio_end"}))

    async def _cleanup(self):
        """Called when session ends — generate visit summary for long-term memory."""
        await self.agent.generate_visit_summary(self.session_id, self.patient_id)
        await self.tts.close()
        logger.info(f"Session {self.session_id} cleaned up")


# ── WebSocket endpoints ───────────────────────────────────────────────────────

@app.websocket("/ws/inbound/{patient_id}")
async def websocket_inbound(websocket: WebSocket, patient_id: str):
    """
    Inbound call endpoint.
    The browser connects here with a patient_id obtained from /api/patients.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"New inbound session {session_id} for patient {patient_id}")

    session = VoiceSession(
        websocket=websocket,
        session_id=session_id,
        patient_id=patient_id,
    )
    await session.run()


@app.websocket("/ws/outbound")
async def websocket_outbound(websocket: WebSocket):
    """
    Outbound call endpoint — Twilio bridges its audio stream here.
    Twilio sends a Start event with the custom parameters we set in TwiML.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    patient_id = None
    campaign_context = None

    # First message from Twilio is a JSON Start event with metadata
    try:
        start_msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        data = json.loads(start_msg)

        if data.get("event") == "start":
            custom_params = data.get("start", {}).get("customParameters", {})
            campaign_context_raw = custom_params.get("campaign_context", "{}")
            campaign_context = json.loads(campaign_context_raw)
            phone = custom_params.get("patient_phone", "")

            # Look up patient from phone
            memory = MemoryManager()
            patient_id = await memory.get_or_create_patient(
                phone=phone,
                name=campaign_context.get("patient_name", "Unknown"),
            )

    except (asyncio.TimeoutError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse Twilio start event: {e}")
        patient_id = str(uuid.uuid4())  # Use a placeholder

    logger.info(f"New outbound session {session_id} for patient {patient_id}")

    session = VoiceSession(
        websocket=websocket,
        session_id=session_id,
        patient_id=patient_id,
        is_outbound=True,
        campaign_context=campaign_context,
    )
    await session.run()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
