# Gemini Live Voice Agent: Real-Time Audio & Non-Blocking Function Calling

A modern, production-grade real-time voice concierge application built on the official **Gemini 3.8 Live API** (`google-genai` SDK) and **FastAPI**, featuring **non-blocking asynchronous function calling**, bidirectional PCM audio streaming, live thinking visualization, and stateful workspace controls.

---

## 🌟 Key Features

1. **True Speech-to-Speech Real-Time Streaming**
   - Microphone capture downsampled in real-time to **16kHz 16-bit PCM** via an `AudioWorkletProcessor`.
   - Live model audio output received at **24kHz 16-bit PCM** and played gaplessly with Web Audio API.
   - Dual-channel stereo visualizer tracking speech activity for both user and agent.

2. **Non-Blocking Background Function Calling (`behavior="NON_BLOCKING"`)**
   - In traditional voice agents, tool execution freezes audio output until the tool returns.
   - With Gemini Live's non-blocking tools, Gemini responds immediately with conversational fillers (e.g., *"Looking up those flights now..."*) while long-running tasks execute concurrently in the background.
   - Supports 5 tools:
     - `search_flights`: Multi-airline schedule and fare lookup.
     - `get_weather_forecast`: Live weather lookup via Open-Meteo API with offline simulation fallback.
     - `deep_research_topic`: Wikipedia-backed multi-angle research briefing.
     - `set_timer_or_reminder`: Real-time countdown timer with state sync.
     - `control_workspace_device`: Smart home device controls (lighting, climate, music, DND).

3. **Dual Model Architecture & Thinking Level Control**
   - **`gemini-3.8-live`**: Default ultra-low latency conversational model with client-side function scheduling (`INTERRUPT`, `WHEN_IDLE`, `SILENT`).
   - **`gemini-3.8-live-extended-thinking`**: Advanced reasoning model with streaming background thoughts, natural filler speech during deliberation, and `interaction_status` (`IN_PROGRESS` vs `IDLE`) lifecycle tracking.
   - Configurable thinking levels: `LOW` (fastest fillers), `MEDIUM` (balanced), `HIGH` (deep analysis).

4. **Instant Interruption Handling**
   - When the user starts speaking while Gemini is talking, Gemini emits `interrupted: true`.
   - The browser audio playback queue is flushed instantly, cancelling speech buffers without latency.
   - Active background tools can be gracefully cancelled on user request (`tool_call_cancellation`).

5. **Session Resumption & Fault Tolerance**
   - Automatic handling of `session_resumption_update` and `go_away` events.
   - Reconnects seamlessly without losing conversation context or active background timers.

---

## 🏗️ Architecture

```
[Browser Client (Chrome / Edge / Firefox)]
   ├── AudioWorkletProcessor (Mic 16kHz PCM Int16 chunking)
   ├── Web Audio API (24kHz Gapless PCM Playback Buffer)
   ├── Dual Visualizer (Stereo Spectrum Canvas)
   └── UI: Dynamic Tool Cards, Transcripts, Thinking Inspector, Workspace State
               │
               ▼ (WebSocket: Direct Binary PCM + JSON Events)
[FastAPI Backend Server (port 8000)]
   ├── /ws Endpoint: Bidirectional WebSocket Bridge
   ├── NonBlockingToolExecutor: Async concurrent task runner
   ├── WorkspaceState: Stateful in-memory smart home environment
   └── google.genai Client: official google-genai SDK (v1beta)
               │
               ▼ (Gemini Live API WebSocket)
[Google Gemini 3.8 Live Engine]
   ├── Speech-to-Speech (Audio In 16kHz -> Audio Out 24kHz)
   ├── Extended Thinking & Conversational Fillers
   └── Non-Blocking Function Declarations (types.Behavior.NON_BLOCKING)
```

---

## 📁 Project Structure

```
gemini-live-voice-agent/
├── backend/
│   ├── __init__.py
│   ├── config.py              # LiveConnectConfig builder & Pydantic settings
│   ├── tools.py               # 5 Non-blocking tool declarations & Async Executor
│   ├── live_session.py        # VoiceAgentSession: Gemini Live WebSocket manager
│   └── main.py                # FastAPI web server and WebSocket endpoint
├── frontend/
│   ├── index.html             # Responsive dark-theme dashboard
│   ├── style.css              # Modern UI layout, glowing visualizers, tool cards
│   ├── app.js                 # WebSocket client, audio playback, event handlers
│   └── audio-worklet-processor.js # Real-time 16kHz PCM downsampler
├── tests/
│   ├── __init__.py
│   ├── test_config.py         # Config validation and model verification tests
│   ├── test_tools.py          # Tool dispatch, async execution, cancellation tests
│   └── test_api.py            # FastAPI REST endpoints tests
├── .env.example               # Environment variables template
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project package configuration & pytest settings
└── README.md                  # Complete documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.11+ installed.
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/)).

### 2. Environment Setup

```bash
cd gemini-live-voice-agent

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

Create `.env` from `.env.example`:

```bash
# Windows:
copy .env.example .env

# Linux / macOS:
cp .env.example .env
```

Edit `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_API_VERSION=v1beta
HOST=127.0.0.1
PORT=8000
```
*(Alternatively, you can leave `.env` empty and enter your API key directly in the browser UI.)*

### 4. Run the Application

```bash
# Using uvicorn:
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 🧪 Running the Automated Test Suite

Run the full pytest suite with 100% async coverage:

```bash
.\.venv\Scripts\pytest.exe -v tests
```

Output:
```
tests/test_api.py::test_health_endpoint PASSED                           [  9%]
tests/test_api.py::test_options_endpoint PASSED                          [ 18%]
tests/test_config.py::test_supported_models_and_voices PASSED            [ 27%]
tests/test_config.py::test_build_config_standard_live PASSED             [ 36%]
tests/test_config.py::test_build_config_extended_thinking PASSED         [ 45%]
tests/test_config.py::test_build_config_session_resumption PASSED        [ 54%]
tests/test_config.py::test_invalid_settings_rejection PASSED             [ 63%]
tests/test_tools.py::test_tool_declarations_non_blocking_behavior PASSED [ 72%]
tests/test_tools.py::test_workspace_state_snapshot PASSED                [ 81%]
tests/test_tools.py::test_tool_executor_weather PASSED                   [ 90%]
tests/test_tools.py::test_tool_executor_cancellation PASSED              [100%]

============================= 11 passed in 5.30s ==============================
```

---

## 🗣️ Voice Commands & Non-Blocking Tool Demos

Once connected, speak naturally with Gemini:

1. **Flight Search (Non-blocking):**
   > *"Aria, can you check flights from San Francisco to Tokyo for next Friday?"*
   - *Gemini immediately speaks filler acknowledgment:* "Looking up flights from SFO to Tokyo right now..."
   - A tool card appears with a progress bar and JSON arguments.
   - You can keep talking while the search runs in the background.

2. **Weather Forecast:**
   > *"What's the weather like in Tashkent today?"*
   - Fetches live temperature, humidity, and condition via Open-Meteo.

3. **In-Depth Research Briefing:**
   > *"Give me a deep research briefing on Quantum Computing focusing on superconducting qubits."*
   - Gemini reasons in the background, outputs thoughts in the Reasoning Inspector, queries references, and speaks the synthesized summary.

4. **Countdown Timer:**
   > *"Set a 30-second tea timer."*
   - Active timer card starts ticking down in real time on the workspace dashboard.

5. **Smart Workspace Controls:**
   > *"Turn off the living room light and set the AC to 20 degrees."*
   - Updates workspace state synchronously on the UI switches without speech lag.

---

## ⚙️ Technical Details: Gemini 3.8 Live API Rules

- **SDK:** Official `google-genai` SDK (`from google import genai; from google.genai import types`). Legacy `google-generativeai` is never used.
- **Audio Format:**
  - Input: Raw 16-bit PCM mono little-endian at 16,000 Hz (`audio/pcm;rate=16000`).
  - Output: Raw 16-bit PCM mono little-endian at 24,000 Hz (`audio/pcm;rate=24000`).
- **Function Declaration:** Every tool specifies `behavior=types.Behavior.NON_BLOCKING`.
- **Response Scheduling:**
  - `INTERRUPT`: Gemini interrupts current speech immediately to speak the function response.
  - `WHEN_IDLE`: Gemini waits until the user and current speech finish before presenting the result.
  - `SILENT`: Gemini consumes the result into memory silently without unsolicited interruption.
- **Session Resumption:** Employs `types.SessionResumptionConfig(handle=...)` to seamlessly survive disconnections and `go_away` events.
