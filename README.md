# Real-Time Multilingual Voice AI Agent — Clinical Appointment Booking

A production-grade voice agent for booking, rescheduling, and cancelling clinical
appointments in English, Hindi, and Tamil. Target end-to-end response latency: **< 450 ms**.

---

## Quick start

### 1. Clone and install

```bash
git clone <your-repo>
cd voice-agent

# Backend
cd backend
cp .env.example .env       # fill in your API keys
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### 2. Start infrastructure

```bash
cd infra
docker-compose up -d       # starts Postgres, Redis, Prometheus
```

### 3. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The server will:
- Create all database tables on first run
- Seed demo doctors and slots
- Start the Prometheus metrics server on port 8001
- Log latency for every request

### 4. Start the frontend

```bash
cd frontend
npm run dev                # http://localhost:3000
```

### 5. (Optional) Outbound campaigns — Celery workers

```bash
cd backend

# Worker: executes call tasks
celery -A scheduler.campaign worker --loglevel=info

# Scheduler: checks for due campaigns every 60s
celery -A scheduler.campaign beat --loglevel=info
```

### 6. (Optional) Expose to Twilio via ngrok

```bash
ngrok http 8000
# Copy the https URL into .env → PUBLIC_BASE_URL
```

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Transport (WebSocket)                         │
│              Browser ↔ FastAPI  /  Twilio ↔ FastAPI             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
     [STT]             [Lang detect]         [TTS]
    Deepgram Nova-2      langdetect         Azure Speech
  Hindi/Tamil/EN       + Deepgram hint    streaming chunks
         │                  │                  ▲
         └──────────────────┼──────────────────┘
                            ▼
              ┌─────────────────────────┐
              │   Agent Orchestrator    │
              │   GPT-4o streaming      │
              │   tool-calling loop     │
              └──────────┬──────────────┘
                         │ calls
          ┌──────────────┼──────────────────┐
          ▼              ▼                  ▼
    get_available   book_appointment   reschedule /
       _slots       (row-locked)       cancel
          │              │                  │
          └──────────────┼──────────────────┘
                         ▼
               ┌──────────────────┐
               │   PostgreSQL     │
               │  doctors, slots  │
               │  appointments    │
               └──────────────────┘

Memory:
  Redis (TTL 30min)  →  session state per call
  PostgreSQL         →  patient profiles + visit summaries
```

---

## Memory design

### Tier 1 — Session memory (Redis)

**Scope**: one conversation. Expires after 30 minutes of inactivity.

**Key**: `session:{session_id}`

**Contents**:
```json
{
  "language": "hi",
  "intent": "book",
  "slots": { "doctor_name": "Dr. Sharma", "date": "2025-01-20" },
  "pending_confirmation": { "slot_id": "...", "display": "Monday 9am with Dr. Sharma" },
  "turn_history": [
    { "role": "user", "content": "mujhe appointment chahiye" },
    { "role": "assistant", "content": "Zaroor! Kaunse doctor ke saath?" }
  ]
}
```

**Why Redis?** Sub-millisecond reads. TTL means stale sessions auto-clean. No
manual lifecycle management needed.

### Tier 2 — Long-term memory (PostgreSQL)

**Scope**: permanent, across all sessions.

**Contents**:
- `patients` table: name, phone, language preference
- `visit_summaries` table: LLM-generated one-sentence summary of each call

**How it's used**: At the start of every session, we load the patient's profile and
their last 3 visit summaries. This is injected into the LLM system prompt as:

```
Patient: Raj Patel | Language preference: Hindi
Past interactions:
  - [2025-01-10] Booked follow-up with Dr. Sharma for 15 Jan.
  - [2025-01-01] Called to reschedule, rebooked for 10 Jan.
```

**How summaries are generated**: At session end, the orchestrator calls GPT-4o with
the full conversation and asks for a one-sentence summary. This is stored async —
it doesn't block the call.

---

## Latency breakdown

Target: **< 450 ms** from speech end to first audio byte.

| Stage | Target | Implementation |
|---|---|---|
| Deepgram STT final | 80 ms | `utterance_end_ms=800`, streaming Nova-2 |
| Redis session load | < 2 ms | In-memory, single key |
| LLM first token | 180 ms | GPT-4o streaming, `max_tokens=300` |
| DB tool call (if triggered) | 30 ms | asyncpg connection pool, indexed queries |
| TTS first chunk | 80 ms | Azure Speech REST, raw PCM stream |
| Network (local) | 10 ms | WebSocket, no serialisation overhead |
| **Total** | **~382 ms** | Well under 450ms target |

### The parallel pipeline trick

The single biggest latency optimisation is **not waiting for the full LLM response
before starting TTS**. The `SentenceBuffer` in `voice/tts.py` watches the LLM
token stream and flushes to Azure Speech as soon as a sentence boundary appears
(`. ! ? । \n`). Azure Speech starts generating audio for sentence 1 while the LLM
is still writing sentence 2.

### Where to see latency logs

Every request logs:
```
[session-id] E2E latency: 387.2ms ✅ UNDER TARGET | STT: 76ms | LLM: 183ms | TTS: 78ms
```

Prometheus metrics are at `http://localhost:8001/metrics`. Key metric:
`voice_agent_e2e_latency_ms` — histogram with 450ms bucket for easy SLA monitoring.

---

## Multilingual handling

- **Detection**: Deepgram Nova-2 detects language automatically per-utterance via
  the `language="multi"` parameter. We also run `langdetect` as a fallback.
- **Persistence**: Detected language is saved to Redis session on first turn.
  For returning patients, their stored language preference from the DB is used.
- **TTS voice selection**: Each language maps to a specific Azure neural voice
  configured in `.env`. The `get_voice_id()` function selects the right voice.
- **LLM prompting**: The system prompt explicitly instructs the LLM to respond in
  the detected language. Hindi and Tamil keywords are included in intent detection.
- **Mid-conversation switching**: If the patient switches languages (e.g. starts
  in English, switches to Hindi), Deepgram detects the new language and the session
  state is updated immediately.

---

## Booking & conflict resolution

All booking operations use PostgreSQL `SELECT FOR UPDATE` row-level locking.
This prevents two concurrent sessions from double-booking the same slot.

Conflict flow:
1. Patient requests slot X
2. Agent calls `book_appointment(slot_id=X)`
3. If X is already booked, the function returns `conflict=True` + a list of
   nearby available alternatives
4. The LLM reads the alternatives and offers them naturally in the patient's language
5. Patient picks one → agent calls `book_appointment` again with the new slot

Validations enforced:
- Past-time slots rejected with a clear message
- Inactive doctors excluded from search results
- Double-booking prevented at DB level (not just application level)

---

## Outbound campaigns

1. Create a campaign via `POST /api/campaigns` with patient IDs and a scheduled time
2. Celery Beat checks every 60 seconds for due campaigns
3. For each patient, a Celery task calls Twilio to place an outbound call
4. Twilio connects the call audio to our `/ws/outbound` WebSocket endpoint
5. The agent plays a greeting, then handles the patient's response naturally
6. Outcome (booked / rescheduled / declined) is written back to `campaign_contacts`

---

## Tradeoffs

| Decision | Why | Alternative |
|---|---|---|
| Deepgram over Whisper | 80ms streaming vs 1-3s batch | Whisper is free but too slow |
| GPT-4o over smaller models | Better tool-calling reliability | Llama-3 via Groq (faster, cheaper, less reliable at tool use) |
| Azure Speech REST over SDK | Simple HTTP deployment path and raw PCM output | SDK can provide lower-level controls |
| Redis session over in-memory | Survives restarts, supports horizontal scaling | In-memory is simpler but not production-safe |
| asyncpg over SQLAlchemy ORM only | Direct SQL for row-locking clarity | ORM alone can obscure locking behaviour |

---

## Known limitations

1. **Tamil TTS quality**: Azure neural Tamil voices are good but can still sound
  less natural for some medical terms; validate with real patient prompts.
2. **Barge-in on mobile Safari**: The Web Audio API's `ScriptProcessorNode` is
   deprecated — a future version should use `AudioWorkletNode`.
3. **No speaker diarization**: Multi-speaker calls (e.g. patient with a family
   member) are not handled — all audio is treated as one speaker.
4. **Campaign scheduler not sharded**: The Celery Beat scheduler is a single
   process. For large campaigns (10,000+ patients), use a distributed beat
   implementation or a dedicated job queue service.
5. **LLM context window**: Very long conversations (30+ turns) will hit the
   `max_tokens` limit on the history. Production should summarise older turns.

---

## Running tests

```bash
cd backend
pytest tests/ -v
```

Requires Docker infrastructure running (`docker-compose up -d`).

---

## Bonus features implemented

- ✅ **Barge-in handling**: Deepgram interim results trigger TTS cancellation
- ✅ **Redis-backed memory with TTL**: `session:{id}` expires after 30 min
- ✅ **Horizontal scalability**: Stateless FastAPI workers; session in Redis
- ✅ **Background job queues**: Celery + Redis for campaign scheduling
- ✅ **Reasoning traces**: Every tool call logged with arguments + result
- ✅ **Prometheus metrics**: E2E, STT, LLM, TTS histograms at `/metrics`
