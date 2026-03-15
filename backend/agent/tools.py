"""
agent/tools.py — Tool definitions for the LLM + dispatcher.

The LLM sees these tool definitions in its system prompt and can
call them when it needs to check availability, book, reschedule, etc.

Each tool:
  1. Has a JSON schema the LLM uses to fill in arguments
  2. Maps to a real async function in scheduler/booking.py
  3. Returns a structured result the LLM reads and responds to

This is genuine tool-calling — NOT simulated/hardcoded responses.
The LLM's reasoning trace will show which tool it called and why.
"""

import json
from typing import Any

from loguru import logger

from scheduler.booking import (
    book_appointment,
    cancel_appointment,
    get_available_slots,
    get_patient_appointments,
    reschedule_appointment,
)
from agent.memory import MemoryManager


# ── Tool schemas (sent to the LLM) ───────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_available_slots",
            "description": (
                "Search for available appointment slots. Use this when the patient asks "
                "about availability, wants to see options, or when you need to find "
                "alternatives after a conflict. You can filter by doctor name, specialty, "
                "or a specific date (ISO format YYYY-MM-DD)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {
                        "type": "string",
                        "description": "Doctor's name (partial match OK). E.g. 'Dr. Sharma'",
                    },
                    "specialty": {
                        "type": "string",
                        "description": "Medical specialty. E.g. 'Cardiologist', 'Dermatologist'",
                    },
                    "date_hint": {
                        "type": "string",
                        "description": "ISO date string YYYY-MM-DD for desired date",
                    },
                    "limit": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "string"},
                        ],
                        "description": "Max number of slots to return (default 5)",
                        "default": 5,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Book a specific appointment slot for the patient. "
                "IMPORTANT: Always confirm with the patient before calling this — "
                "describe the slot and ask 'Shall I confirm this booking?' first. "
                "Only call this tool after the patient says yes/haan/aamaam (yes in Hindi/Tamil)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "The patient's UUID from their profile",
                    },
                    "slot_id": {
                        "type": "string",
                        "description": "The UUID of the specific slot to book",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for the visit (optional)",
                    },
                },
                "required": ["patient_id", "slot_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": (
                "Reschedule an existing appointment to a new slot. "
                "First use get_available_slots to find options, present them to the patient, "
                "then call this after they confirm."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type": "string",
                        "description": "UUID of the existing appointment to reschedule",
                    },
                    "new_slot_id": {
                        "type": "string",
                        "description": "UUID of the new slot",
                    },
                },
                "required": ["appointment_id", "new_slot_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": (
                "Cancel an existing appointment. Always confirm with the patient first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {
                        "type": "string",
                        "description": "UUID of the appointment to cancel",
                    },
                },
                "required": ["appointment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_appointments",
            "description": "Retrieve the patient's upcoming confirmed appointments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "The patient's UUID",
                    },
                },
                "required": ["patient_id"],
            },
        },
    },
]


# ── Dispatcher ────────────────────────────────────────────────────────────────

class ToolDispatcher:
    """Maps tool names to their implementations and executes them."""

    def __init__(self):
        self._registry = {
            "get_available_slots": get_available_slots,
            "book_appointment": book_appointment,
            "reschedule_appointment": reschedule_appointment,
            "cancel_appointment": cancel_appointment,
            "get_patient_appointments": get_patient_appointments,
        }

    @staticmethod
    def _normalize_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Coerce common LLM argument mismatches to expected tool types."""
        normalized = dict(arguments)

        if tool_name == "get_available_slots":
            limit = normalized.get("limit")
            if isinstance(limit, str):
                stripped = limit.strip()
                if stripped.isdigit():
                    normalized["limit"] = int(stripped)

        return normalized

    async def dispatch(self, tool_name: str, arguments: dict) -> str:
        """
        Execute a tool and return its result as a JSON string.
        The JSON string is added back to the conversation so the LLM can read it.
        """
        func = self._registry.get(tool_name)
        if not func:
            logger.error(f"Unknown tool: {tool_name}")
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        logger.info(f"Tool call: {tool_name}({json.dumps(arguments, default=str)})")
        try:
            normalized_args = self._normalize_arguments(tool_name, arguments)
            result = await func(**normalized_args)
            logger.info(f"Tool result: {json.dumps(result, default=str)[:200]}")
            return json.dumps(result, default=str)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return json.dumps({"error": str(e)})
