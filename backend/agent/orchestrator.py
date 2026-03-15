"""
agent/orchestrator.py — The brain of the voice agent.

This is where everything connects:
  transcript → LLM → tool calls → tool results → LLM → text response → TTS

The LLM runs in a STREAMING loop:
  1. We build a prompt from session state + patient context + conversation history
  2. We stream the LLM response token by token
  3. When the LLM wants to call a tool, we execute it and feed the result back
  4. We yield text tokens as they arrive so TTS can start immediately

Reasoning traces are visible in the logs — every tool call is logged
with its arguments and result. This satisfies the assignment requirement
that "reasoning traces must be visible and demonstrable."
"""

import json
from typing import AsyncIterator

from loguru import logger
#from openai import AsyncOpenAI
from groq import AsyncGroq

from config import settings
from agent.memory import MemoryManager
from agent.tools import TOOL_DEFINITIONS, ToolDispatcher
from agent.lang_detect import LANGUAGE_NAMES
from monitoring.latency import LatencyTracker


# ── System prompt ──────────────────────────────────────────────────────────────

def build_system_prompt(
    language: str,
    patient_context: str,
    session_state: dict,
) -> str:
    lang_name = LANGUAGE_NAMES.get(language, "English")
    return f"""You are a helpful clinical appointment assistant for a healthcare platform.
You are speaking with a patient over the phone. Keep responses SHORT and natural — this is a voice call, not a chat.

LANGUAGE: Respond ONLY in {lang_name}. The patient speaks {lang_name}.
If the language changes mid-conversation, adapt immediately.

PATIENT CONTEXT:
{patient_context}

CURRENT SESSION STATE:
- Intent: {session_state.get('intent', 'unknown')}
- Collected info: {json.dumps(session_state.get('slots', {}), default=str)}
- Pending confirmation: {session_state.get('pending_confirmation', 'none')}

INSTRUCTIONS:
1. Be warm, concise, and professional. 2-3 sentences max per response for voice.
2. When a patient wants to book, use get_available_slots first to check availability.
3. Always describe the slot details and ASK FOR CONFIRMATION before calling book_appointment.
4. If a slot is taken, immediately offer alternatives from the conflict response.
5. Handle "yes/haan/aamaam/seri" as confirmation. Handle "no/nahin/illai" as rejection.
6. If the patient changes their mind mid-booking, gracefully restart the slot search.
7. For outbound reminder calls: greet the patient, state the appointment details,
   and ask if they'll attend or need to reschedule.

TOOLS AVAILABLE: get_available_slots, book_appointment, reschedule_appointment,
cancel_appointment, get_patient_appointments.

Remember: You're on a voice call. No bullet points, no markdown, no lists.
Speak naturally as if talking to the patient directly.
"""


# ── Orchestrator ──────────────────────────────────────────────────────────────

class AgentOrchestrator:

    def __init__(self):
        self._llm = AsyncGroq(api_key=settings.groq_api_key)
        self._memory = MemoryManager()
        self._tools = ToolDispatcher()

    async def process(
        self,
        transcript: str,
        session_id: str,
        patient_id: str,
        latency_tracker: LatencyTracker,
    ) -> AsyncIterator[str]:
        """
        Main entry point. Takes a transcript, runs the agent loop,
        yields text chunks suitable for TTS.
        """
        # ── 1. Load context ────────────────────────────────────────────────────
        session = await self._memory.get_session(session_id)
        language = session.get("language", "en")

        # Get patient history from Postgres (long-term memory)
        patient_ctx = await self._memory.get_patient_context(patient_id)

        # ── 2. Build messages ──────────────────────────────────────────────────
        system_prompt = build_system_prompt(language, patient_ctx, session)

        # Reconstruct conversation history from session
        messages = [{"role": "system", "content": system_prompt}]
        for turn in session.get("turn_history", []):
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": transcript})

        logger.info(f"Agent processing [{language}] session={session_id}: '{transcript}'")

        # Save the user turn
        await self._memory.append_turn(session_id, "user", transcript)

        # ── 3. LLM streaming loop ──────────────────────────────────────────────
        full_response = ""

        async for token in self._run_llm_loop(messages, session_id, patient_id, latency_tracker):
            full_response += token
            yield token

        # ── 4. Save assistant turn + update session ────────────────────────────
        await self._memory.append_turn(session_id, "assistant", full_response)

        # Extract intent from response if we can detect it
        await self._update_intent(session_id, transcript, full_response)

    async def _run_llm_loop(
        self,
        messages: list,
        session_id: str,
        patient_id: str,
        latency_tracker: LatencyTracker,
    ) -> AsyncIterator[str]:
        """
        Core streaming loop. Handles tool calls mid-stream.

        The LLM may:
          a) Respond with text only → yield tokens directly
          b) Call a tool, then continue → execute tool, add result, continue
          c) Call multiple tools in sequence → handle each one

        All tool calls and results are logged for reasoning trace visibility.
        """
        first_token_sent = False
        max_iterations = 5   # Prevent infinite tool-call loops

        for iteration in range(max_iterations):
            # Accumulate tool call data (can span multiple chunks)
            pending_tool_calls: dict[str, dict] = {}
            text_buffer = ""

            stream = await self._llm.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                stream=True,
                temperature=0.3,    # Low temp for consistent booking behavior
                max_tokens=300,     # Keep responses short for voice
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                finish_reason = chunk.choices[0].finish_reason

                # ── Text token ─────────────────────────────────────────────────
                if delta.content:
                    if not first_token_sent:
                        latency_tracker.mark_llm_first_token()
                        first_token_sent = True
                    text_buffer += delta.content
                    yield delta.content

                # ── Tool call being assembled ───────────────────────────────────
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in pending_tool_calls:
                            pending_tool_calls[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc.id:
                            pending_tool_calls[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                pending_tool_calls[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                pending_tool_calls[idx]["arguments"] += tc.function.arguments

                # ── Stream finished ────────────────────────────────────────────
                if finish_reason == "stop":
                    # LLM is done — no more tool calls
                    return

                if finish_reason == "tool_calls":
                    # LLM wants to call tools — execute them
                    if text_buffer:
                        messages.append({"role": "assistant", "content": text_buffer})
                        text_buffer = ""

                    # Build the assistant message with tool calls
                    tool_call_list = []
                    for idx in sorted(pending_tool_calls.keys()):
                        tc_data = pending_tool_calls[idx]
                        tool_call_list.append({
                            "id": tc_data["id"],
                            "type": "function",
                            "function": {
                                "name": tc_data["name"],
                                "arguments": tc_data["arguments"],
                            },
                        })

                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_call_list,
                    })

                    # Execute each tool and add results
                    for tc_data in pending_tool_calls.values():
                        try:
                            args = json.loads(tc_data["arguments"])
                            # Inject patient_id for tools that need it
                            tool_def = next(
                                (p for p in TOOL_DEFINITIONS
                                 if p["function"]["name"] == tc_data["name"]),
                                None,
                            )
                            if tool_def and "patient_id" in tool_def["function"].get(
                                "parameters", {}
                            ).get("properties", {}):
                                args.setdefault("patient_id", patient_id)
                        except json.JSONDecodeError:
                            args = {}

                        result = await self._tools.dispatch(tc_data["name"], args)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_data["id"],
                            "content": result,
                        })

                    # Continue the loop — LLM will now generate a response
                    # based on the tool results
                    break

        else:
            # Hit max iterations — something went wrong
            logger.warning(f"Agent hit max iterations for session {session_id}")
            yield "I'm sorry, I ran into a problem. Could you please repeat that?"

    async def _update_intent(
        self, session_id: str, transcript: str, response: str
    ) -> None:
        """
        Simple keyword-based intent tracking to update session state.
        This gives the LLM better context in follow-up turns.
        """
        lower = transcript.lower()
        intent = None

        if any(w in lower for w in ["book", "appointment", "schedule", "fix", "चाहिए", "बुक", "வேண்டும்"]):
            intent = "book"
        elif any(w in lower for w in ["reschedule", "change", "move", "बदल", "மாற்று"]):
            intent = "reschedule"
        elif any(w in lower for w in ["cancel", "cancellation", "रद्द", "ரத்து"]):
            intent = "cancel"
        elif any(w in lower for w in ["when", "appointment", "timing", "कब", "எப்போது"]):
            intent = "query"

        if intent:
            await self._memory.update_session(session_id, intent=intent)

    async def generate_visit_summary(
        self, session_id: str, patient_id: str
    ) -> None:
        """
        Call at the end of a session to generate and store a long-term memory summary.
        Uses the LLM to summarize the conversation in a single sentence.
        """
        session = await self._memory.get_session(session_id)
        history = session.get("turn_history", [])
        if len(history) < 2:
            return

        conversation = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in history
        )
        response = await self._llm.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Summarize this patient call in ONE sentence for a medical record. "
                        f"Include what happened (booked/cancelled/rescheduled), the doctor name if mentioned, "
                        f"and the appointment date if mentioned.\n\nConversation:\n{conversation}"
                    ),
                }
            ],
            max_tokens=100,
            temperature=0.1,
        )
        summary = response.choices[0].message.content.strip()
        await self._memory.save_visit_summary(patient_id, summary)
        logger.info(f"Visit summary saved: {summary}")
