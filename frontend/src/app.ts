/**
 * app.ts — Main frontend application.
 *
 * Flow:
 *   1. User enters their phone number → we create/look up their patient record
 *   2. We open a WebSocket to /ws/inbound/{patient_id}
 *   3. AudioCapture streams mic audio → WebSocket → Deepgram STT
 *   4. Backend sends back transcript events (JSON) and audio bytes
 *   5. AudioPlayer plays the audio; barge-in stops it immediately
 */

import { AudioCapture } from "./audioCapture";
import { AudioPlayer } from "./audioPlayer";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
const WS_URL = BACKEND_URL.replace("http", "ws");

// ── State ──────────────────────────────────────────────────────────────────

let ws: WebSocket | null = null;
let capture: AudioCapture | null = null;
let player: AudioPlayer | null = null;
let patientId: string | null = null;
let isConnected = false;

// ── DOM refs ───────────────────────────────────────────────────────────────

const phoneInput = document.getElementById("phone") as HTMLInputElement;
const nameInput = document.getElementById("name") as HTMLInputElement;
const connectBtn = document.getElementById("connect-btn") as HTMLButtonElement;
const disconnectBtn = document.getElementById("disconnect-btn") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLDivElement;
const transcriptEl = document.getElementById("transcript") as HTMLDivElement;
const logEl = document.getElementById("log") as HTMLDivElement;

// ── Helpers ────────────────────────────────────────────────────────────────

function setStatus(text: string, color: "green" | "red" | "orange" = "green") {
  statusEl.textContent = text;
  statusEl.style.color = { green: "#0f6e56", red: "#993c1d", orange: "#854f0b" }[color];
}

function addLog(text: string) {
  const line = document.createElement("div");
  line.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
  logEl.prepend(line);
}

function addTranscript(role: "patient" | "agent", text: string, lang: string) {
  const el = document.createElement("div");
  el.className = `turn ${role}`;
  el.innerHTML = `<span class="role">${role === "patient" ? "🧑 You" : "🤖 Agent"}</span>
                  <span class="lang">[${lang.toUpperCase()}]</span>
                  <span class="text">${text}</span>`;
  transcriptEl.prepend(el);
}

// ── Patient registration ───────────────────────────────────────────────────

async function registerPatient(): Promise<string> {
  const res = await fetch(`${BACKEND_URL}/api/patients`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      phone: phoneInput.value.trim(),
      name: nameInput.value.trim() || "Unknown",
    }),
  });
  if (!res.ok) throw new Error("Failed to register patient");
  const data = await res.json();
  return data.patient_id;
}

// ── WebSocket session ──────────────────────────────────────────────────────

async function connect() {
  connectBtn.disabled = true;
  setStatus("Connecting...", "orange");

  try {
    patientId = await registerPatient();
    addLog(`Patient ID: ${patientId}`);

    // Open WebSocket
    ws = new WebSocket(`${WS_URL}/ws/inbound/${patientId}`);
    ws.binaryType = "arraybuffer";

    ws.onopen = async () => {
      isConnected = true;
      setStatus("Connected — speak now", "green");
      disconnectBtn.disabled = false;
      addLog("WebSocket connected");

      // Start capturing microphone
      player = new AudioPlayer();
      capture = new AudioCapture((chunk) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(chunk);
        }
      });
      await capture.start();
      addLog("Microphone active");
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        // Binary message = audio bytes from TTS → play it
        player?.enqueue(event.data);
      } else {
        // Text message = JSON control event
        handleControlMessage(JSON.parse(event.data));
      }
    };

    ws.onclose = () => {
      isConnected = false;
      setStatus("Disconnected", "red");
      addLog("WebSocket closed");
      resetUI();
    };

    ws.onerror = (err) => {
      addLog(`WebSocket error: ${err}`);
      setStatus("Connection error", "red");
    };

  } catch (err) {
    setStatus(`Error: ${err}`, "red");
    connectBtn.disabled = false;
  }
}

function handleControlMessage(msg: Record<string, unknown>) {
  switch (msg.type) {
    case "transcript":
      // Patient's words recognised by STT
      addTranscript("patient", msg.text as string, msg.language as string);
      addLog(`STT [${msg.language}]: "${msg.text}"`);
      break;

    case "agent_text":
      // Agent's response text (for display)
      addTranscript("agent", msg.text as string, msg.language as string);
      break;

    case "barge_in":
      // Patient spoke while agent was talking — stop audio immediately
      player?.stop();
      addLog("Barge-in detected — stopped playback");
      break;

    case "audio_end":
      // Agent finished speaking
      addLog("Agent finished speaking");
      break;

    default:
      addLog(`Event: ${JSON.stringify(msg)}`);
  }
}

function disconnect() {
  capture?.stop();
  ws?.close();
  resetUI();
}

function resetUI() {
  connectBtn.disabled = false;
  disconnectBtn.disabled = true;
  capture = null;
  player = null;
}

// ── Event listeners ────────────────────────────────────────────────────────

connectBtn.addEventListener("click", connect);
disconnectBtn.addEventListener("click", disconnect);
