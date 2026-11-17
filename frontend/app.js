/**
 * Gemini Live Voice Agent - Client Application
 * Full-duplex binary PCM audio streaming (16kHz in, 24kHz out),
 * instant interruption cancellation, non-blocking tool cards,
 * live transcript, thought reasoning stream, and workspace state sync.
 */

// State
let ws = null;
let audioContext = null;
let micStream = null;
let workletNode = null;
let playbackContext = null;
let nextPlayTime = 0;
let activeAudioSources = [];
let isConnected = false;
let isMuted = false;

// Analysers & Visualizer
let micAnalyser = null;
let playbackAnalyser = null;
let visualizerAnimId = null;

// UI Elements
const connectBtn = document.getElementById('connectBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const muteBtn = document.getElementById('muteBtn');
const interruptBtn = document.getElementById('interruptBtn');
const modelSelect = document.getElementById('modelSelect');
const voiceSelect = document.getElementById('voiceSelect');
const thinkingLevelGroup = document.getElementById('thinkingLevelGroup');
const thinkingSelect = document.getElementById('thinkingSelect');
const apiKeyInput = document.getElementById('apiKeyInput');
const statusBadge = document.getElementById('statusBadge');
const interactionBadge = document.getElementById('interactionBadge');
const resumeBadge = document.getElementById('resumeBadge');
const transcriptContainer = document.getElementById('transcriptContainer');
const thoughtContainer = document.getElementById('thoughtContainer');
const thoughtBox = document.getElementById('thoughtBox');
const activeToolsList = document.getElementById('activeToolsList');
const textInput = document.getElementById('textInput');
const sendTextBtn = document.getElementById('sendTextBtn');
const canvas = document.getElementById('visualizerCanvas');
const canvasCtx = canvas ? canvas.getContext('2d') : null;

// Workspace Elements
const livingRoomLight = document.getElementById('livingRoomLight');
const studyLight = document.getElementById('studyLight');
const tempDisplay = document.getElementById('tempDisplay');
const musicStatus = document.getElementById('musicStatus');
const musicTrack = document.getElementById('musicTrack');
const dndStatus = document.getElementById('dndStatus');
const activeTimersList = document.getElementById('activeTimersList');

// Tool Schedulers
const schedFlight = document.getElementById('schedFlight');
const schedWeather = document.getElementById('schedWeather');
const schedResearch = document.getElementById('schedResearch');
const schedTimer = document.getElementById('schedTimer');
const schedDevice = document.getElementById('schedDevice');

// Initialize local storage API key
const savedApiKey = localStorage.getItem('gemini_api_key');
if (savedApiKey && apiKeyInput) {
  apiKeyInput.value = savedApiKey;
}
if (apiKeyInput) {
  apiKeyInput.addEventListener('input', () => {
    localStorage.setItem('gemini_api_key', apiKeyInput.value.trim());
  });
}

// Toggle thinking level visibility based on model
function updateModelUI() {
  const isExtended = modelSelect.value.includes('extended-thinking');
  if (thinkingLevelGroup) {
    thinkingLevelGroup.style.display = isExtended ? 'block' : 'none';
  }
}
if (modelSelect) {
  modelSelect.addEventListener('change', updateModelUI);
  updateModelUI();
}

/**
 * Initialize Web Audio API contexts
 */
async function initAudio() {
  // Capture Context (Mic)
  audioContext = new (window.AudioContext || window.webkitAudioContext)({
    latencyHint: 'interactive'
  });
  await audioContext.resume();

  // Playback Context at 24kHz for Gemini Live PCM output
  playbackContext = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: 24000,
    latencyHint: 'interactive'
  });
  await playbackContext.resume();
  nextPlayTime = playbackContext.currentTime;

  // Analysers for stereo spectrum visualizer
  micAnalyser = audioContext.createAnalyser();
  micAnalyser.fftSize = 64;

  playbackAnalyser = playbackContext.createAnalyser();
  playbackAnalyser.fftSize = 64;

  // Load Worklet Processor
  await audioContext.audioWorklet.addModule('/static/audio-worklet-processor.js');

  // Request Microphone Stream
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: { ideal: 16000 },
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true
    }
  });

  const micSource = audioContext.createMediaStreamSource(micStream);
  micSource.connect(micAnalyser);

  workletNode = new AudioWorkletNode(audioContext, 'pcm-recorder-processor');
  micSource.connect(workletNode);

  // Send binary 16kHz PCM chunks straight over WebSocket
  workletNode.port.onmessage = (event) => {
    if (!isConnected || isMuted || !ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }
    // Direct binary transfer
    ws.send(event.data);
  };

  startVisualizer();
}

/**
 * Playback 24kHz 16-bit PCM binary chunk
 */
function playPcmBinary(arrayBuffer) {
  if (!playbackContext) return;

  const int16Array = new Int16Array(arrayBuffer);
  const numSamples = int16Array.length;
  if (numSamples === 0) return;

  const audioBuffer = playbackContext.createBuffer(1, numSamples, 24000);
  const channelData = audioBuffer.getChannelData(0);

  for (let i = 0; i < numSamples; i++) {
    const s = int16Array[i];
    channelData[i] = s < 0 ? s / 0x8000 : s / 0x7FFF;
  }

  const sourceNode = playbackContext.createBufferSource();
  sourceNode.buffer = audioBuffer;
  sourceNode.connect(playbackAnalyser);
  playbackAnalyser.connect(playbackContext.destination);

  const now = playbackContext.currentTime;
  if (nextPlayTime < now) {
    nextPlayTime = now + 0.01;
  }

  sourceNode.start(nextPlayTime);
  nextPlayTime += audioBuffer.duration;

  activeAudioSources.push(sourceNode);
  sourceNode.onended = () => {
    const idx = activeAudioSources.indexOf(sourceNode);
    if (idx !== -1) {
      activeAudioSources.splice(idx, 1);
    }
  };
}

/**
 * Immediate interruption handling: stop playing audio & flush buffers
 */
function handleInterruption() {
  console.log('[Audio] Interrupted by speech! Flushing audio playback buffer.');
  for (const src of activeAudioSources) {
    try {
      src.stop();
      src.disconnect();
    } catch (e) {}
  }
  activeAudioSources = [];
  if (playbackContext) {
    nextPlayTime = playbackContext.currentTime;
  }
  appendTranscript('system', '⚡ [Gemini interrupted by speech]');
}

/**
 * Connect WebSocket & Start Session
 */
async function connect() {
  try {
    updateStatus('Connecting...', 'connecting');
    await initAudio();

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer'; // Crucial for direct binary PCM

    ws.onopen = () => {
      console.log('[WS] Connected to FastAPI backend');
      isConnected = true;
      connectBtn.disabled = true;
      disconnectBtn.disabled = false;
      muteBtn.disabled = false;
      interruptBtn.disabled = false;
      sendTextBtn.disabled = false;

      // First frame MUST be "start" with settings
      const startPayload = {
        type: 'start',
        settings: {
          api_key: apiKeyInput.value.trim() || null,
          model: modelSelect.value,
          voice_name: voiceSelect.value,
          thinking_level: thinkingSelect.value,
          tool_scheduling: {
            search_flights: schedFlight.value,
            get_weather_forecast: schedWeather.value,
            deep_research_topic: schedResearch.value,
            set_timer_or_reminder: schedTimer.value,
            control_workspace_device: schedDevice.value
          }
        }
      };

      ws.send(JSON.stringify(startPayload));
      updateStatus('Initializing Session...', 'connecting');
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        // Binary 24kHz PCM from Gemini
        playPcmBinary(event.data);
      } else {
        // JSON event
        try {
          const msg = JSON.parse(event.data);
          handleServerEvent(msg);
        } catch (err) {
          console.error('[WS] Failed to parse message:', err, event.data);
        }
      }
    };

    ws.onerror = (err) => {
      console.error('[WS] WebSocket error:', err);
      updateStatus('Connection Error', 'error');
    };

    ws.onclose = () => {
      console.log('[WS] WebSocket closed');
      disconnect();
    };

  } catch (err) {
    console.error('[App] Failed to start:', err);
    updateStatus('Error: ' + err.message, 'error');
    disconnect();
  }
}

/**
 * Handle server JSON telemetry & lifecycle events
 */
function handleServerEvent(msg) {
  switch (msg.type) {
    case 'session_ready':
      updateStatus(`Live (${msg.model})`, 'connected');
      appendTranscript('system', `✨ Session established! Model: <strong>${msg.model}</strong> (VAD: ${msg.vad_mode})`);
      break;

    case 'session_reconnecting':
      updateStatus(`Reconnecting (${msg.attempt})...`, 'connecting');
      appendTranscript('system', `🔄 Reconnecting to Gemini Live (attempt ${msg.attempt})...`);
      break;

    case 'interrupted':
      handleInterruption();
      break;

    case 'input_transcription':
      appendTranscript('user', msg.text);
      break;

    case 'output_transcription':
    case 'model_text':
      appendTranscript('model', msg.text);
      break;

    case 'thought_summary':
      appendThought(msg.text);
      break;

    case 'interaction_status':
      if (interactionBadge) {
        interactionBadge.textContent = msg.status;
        interactionBadge.className = `badge badge-${msg.status.toLowerCase()}`;
      }
      break;

    case 'turn_complete':
      markTurnComplete();
      break;

    case 'tool_started':
      renderToolStarted(msg.tool);
      break;

    case 'tool_progress':
      updateToolProgress(msg.tool);
      break;

    case 'tool_completed':
      renderToolCompleted(msg.tool);
      break;

    case 'tool_cancelled':
      renderToolCancelled(msg.tool);
      break;

    case 'workspace_state':
      updateWorkspaceUI(msg.state);
      break;

    case 'resumption_handle':
      if (resumeBadge) {
        resumeBadge.textContent = 'Session Resumable';
        resumeBadge.style.display = 'inline-block';
      }
      break;

    case 'go_away':
      appendTranscript('system', `⚠️ Server signaled session rollover (time left: ${msg.time_left}). Auto-resuming...`);
      break;

    case 'error':
      appendTranscript('system', `❌ Error: ${msg.message}`);
      break;
  }
}

/**
 * Clean disconnection
 */
function disconnect() {
  isConnected = false;
  updateStatus('Disconnected', 'disconnected');

  if (ws) {
    if (ws.readyState === WebSocket.OPEN) {
      try { ws.send(JSON.stringify({ type: 'stop' })); } catch (e) {}
    }
    try { ws.close(); } catch (e) {}
    ws = null;
  }

  handleInterruption();

  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }

  if (audioContext) {
    try { audioContext.close(); } catch (e) {}
    audioContext = null;
  }

  if (playbackContext) {
    try { playbackContext.close(); } catch (e) {}
    playbackContext = null;
  }

  if (visualizerAnimId) {
    cancelAnimationFrame(visualizerAnimId);
    visualizerAnimId = null;
  }

  connectBtn.disabled = false;
  disconnectBtn.disabled = true;
  muteBtn.disabled = true;
  interruptBtn.disabled = true;
  sendTextBtn.disabled = true;
}

/**
 * Transcripts Management
 */
let currentTranscriptEl = null;
let currentTranscriptSource = null;

function appendTranscript(source, text) {
  if (!transcriptContainer || !text) return;

  if (source === 'system') {
    const el = document.createElement('div');
    el.className = 'transcript-item system-msg';
    el.innerHTML = text;
    transcriptContainer.appendChild(el);
    transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
    currentTranscriptEl = null;
    return;
  }

  if (currentTranscriptEl && currentTranscriptSource === source) {
    currentTranscriptEl.querySelector('.text-body').textContent += text;
  } else {
    currentTranscriptSource = source;
    currentTranscriptEl = document.createElement('div');
    currentTranscriptEl.className = `transcript-item ${source}-msg`;

    const header = document.createElement('div');
    header.className = 'speaker-label';
    header.textContent = source === 'user' ? '👤 You' : '🤖 Gemini';

    const body = document.createElement('div');
    body.className = 'text-body';
    body.textContent = text;

    currentTranscriptEl.appendChild(header);
    currentTranscriptEl.appendChild(body);
    transcriptContainer.appendChild(currentTranscriptEl);
  }

  transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
}

function markTurnComplete() {
  currentTranscriptEl = null;
  currentTranscriptSource = null;
}

/**
 * Thought Stream Management
 */
function appendThought(text) {
  if (!thoughtBox || !thoughtContainer || !text) return;
  thoughtBox.style.display = 'flex';
  thoughtContainer.textContent += text;
  thoughtContainer.scrollTop = thoughtContainer.scrollHeight;
}

/**
 * Non-Blocking Tool Execution Card UI
 */
function renderToolStarted(tool) {
  if (!activeToolsList) return;

  const emptyState = activeToolsList.querySelector('.empty-tools-state');
  if (emptyState) emptyState.remove();

  let card = document.getElementById(`tool-${tool.id}`);
  if (!card) {
    card = document.createElement('div');
    card.id = `tool-${tool.id}`;
    activeToolsList.prepend(card);
  }

  card.className = 'tool-card running';
  card.innerHTML = `
    <div class="tool-card-header">
      <div class="tool-name">
        <span class="spinner-small"></span>
        <strong>${escapeHtml(tool.name)}</strong>
      </div>
      <span class="badge badge-info">${escapeHtml(tool.scheduling || 'WHEN_IDLE')}</span>
    </div>
    <div class="tool-args"><code>${escapeHtml(JSON.stringify(tool.args, null, 2))}</code></div>
    <div class="tool-progress-bar-container">
      <div class="tool-progress-bar" style="width: 25%"></div>
    </div>
    <div class="tool-status-text">${escapeHtml(tool.progress || 'Executing in background...')}</div>
  `;
}

function updateToolProgress(tool) {
  const card = document.getElementById(`tool-${tool.id}`);
  if (!card) return;

  const text = card.querySelector('.tool-status-text');
  if (text && tool.progress) {
    text.textContent = tool.progress;
  }
}

function renderToolCompleted(tool) {
  const card = document.getElementById(`tool-${tool.id}`);
  if (!card) return;

  card.className = 'tool-card completed';
  const spinner = card.querySelector('.spinner-small');
  if (spinner) spinner.remove();

  const bar = card.querySelector('.tool-progress-bar');
  if (bar) bar.style.width = '100%';

  const durationSec = tool.duration_ms ? (tool.duration_ms / 1000).toFixed(2) : '0.5';
  const statusText = card.querySelector('.tool-status-text');
  if (statusText) {
    statusText.innerHTML = `✅ Completed in ${durationSec}s`;
  }

  const resultContainer = document.createElement('div');
  resultContainer.className = 'tool-result';
  resultContainer.innerHTML = `<details><summary>Output Payload</summary><pre><code>${escapeHtml(JSON.stringify(tool.result, null, 2))}</code></pre></details>`;
  card.appendChild(resultContainer);
}

function renderToolCancelled(tool) {
  const card = document.getElementById(`tool-${tool.id}`);
  if (!card) return;

  card.className = 'tool-card cancelled';
  const spinner = card.querySelector('.spinner-small');
  if (spinner) spinner.remove();

  const statusText = card.querySelector('.tool-status-text');
  if (statusText) {
    statusText.innerHTML = `🚫 <em>${escapeHtml(tool.progress || 'Cancelled on interruption')}</em>`;
  }
}

/**
 * Smart Workspace State UI Update
 */
function updateWorkspaceUI(state) {
  if (!state) return;

  if (livingRoomLight && state.lights) {
    livingRoomLight.checked = state.lights.on;
  }

  if (tempDisplay && state.thermostat_c !== undefined) {
    tempDisplay.textContent = `${state.thermostat_c}°C (auto)`;
  }

  if (musicStatus && state.music) {
    musicStatus.textContent = state.music.playing ? 'Playing' : 'Paused';
  }
  if (musicTrack && state.music) {
    musicTrack.textContent = state.music.playing ? `Genre: ${state.music.genre}` : 'None';
  }

  if (dndStatus && state.do_not_disturb !== undefined) {
    dndStatus.textContent = state.do_not_disturb ? 'ON' : 'OFF';
    dndStatus.className = state.do_not_disturb ? 'badge badge-danger' : 'badge badge-secondary';
  }

  if (activeTimersList && state.timers) {
    activeTimersList.innerHTML = '';
    const timers = state.timers;
    if (timers.length === 0) {
      activeTimersList.innerHTML = '<li class="text-muted">No active timers</li>';
    } else {
      for (const t of timers) {
        const li = document.createElement('li');
        li.innerHTML = `<strong>${escapeHtml(t.label)}</strong>: ${t.remaining_seconds}s (${t.status})`;
        activeTimersList.appendChild(li);
      }
    }
  }
}

/**
 * Workspace device direct control from UI
 */
window.toggleDevice = function(device, key, value) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  // Trigger text event as client content
  const actionText = value ? `Turn on ${device.replace('_', ' ')}` : `Turn off ${device.replace('_', ' ')}`;
  ws.send(JSON.stringify({
    type: 'client_content',
    text: actionText,
    turn_complete: true
  }));
};

/**
 * Text Chat Submission
 */
function sendText() {
  if (!textInput || !ws || ws.readyState !== WebSocket.OPEN) return;
  const text = textInput.value.trim();
  if (!text) return;

  ws.send(JSON.stringify({
    type: 'client_content',
    text: text,
    turn_complete: true
  }));

  appendTranscript('user', text);
  textInput.value = '';
}

if (sendTextBtn) sendTextBtn.addEventListener('click', sendText);
if (textInput) {
  textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendText();
  });
}

/**
 * Stereo Frequency Visualizer
 */
function startVisualizer() {
  if (!canvas || !canvasCtx) return;

  const micData = new Uint8Array(micAnalyser ? micAnalyser.frequencyBinCount : 32);
  const playData = new Uint8Array(playbackAnalyser ? playbackAnalyser.frequencyBinCount : 32);

  function draw() {
    visualizerAnimId = requestAnimationFrame(draw);
    const width = canvas.width;
    const height = canvas.height;

    canvasCtx.clearRect(0, 0, width, height);

    if (micAnalyser) micAnalyser.getByteFrequencyData(micData);
    if (playbackAnalyser) playbackAnalyser.getByteFrequencyData(playData);

    const barCount = 20;
    const barWidth = (width / 2) / barCount - 2;

    // Mic (User) in Cyan
    for (let i = 0; i < barCount; i++) {
      const val = micData[i] || 0;
      const barH = (val / 255) * (height - 10);
      canvasCtx.fillStyle = isMuted ? '#555' : '#00e5ff';
      canvasCtx.fillRect(width / 2 - (i + 1) * (barWidth + 2), height / 2 - barH / 2, barWidth, Math.max(3, barH));
    }

    // Gemini (Playback) in Purple
    for (let i = 0; i < barCount; i++) {
      const val = playData[i] || 0;
      const barH = (val / 255) * (height - 10);
      canvasCtx.fillStyle = '#b388ff';
      canvasCtx.fillRect(width / 2 + i * (barWidth + 2) + 2, height / 2 - barH / 2, barWidth, Math.max(3, barH));
    }

    // Divider
    canvasCtx.fillStyle = '#333';
    canvasCtx.fillRect(width / 2 - 1, 4, 2, height - 8);
  }

  draw();
}

/**
 * Dynamic Tool Scheduling Update
 */
function updateToolScheduling(toolName, schedValue) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'update_scheduling',
      tool_name: toolName,
      scheduling: schedValue
    }));
  }
}

if (schedFlight) schedFlight.addEventListener('change', () => updateToolScheduling('search_flights', schedFlight.value));
if (schedWeather) schedWeather.addEventListener('change', () => updateToolScheduling('get_weather_forecast', schedWeather.value));
if (schedResearch) schedResearch.addEventListener('change', () => updateToolScheduling('deep_research_topic', schedResearch.value));
if (schedTimer) schedTimer.addEventListener('change', () => updateToolScheduling('set_timer_or_reminder', schedTimer.value));
if (schedDevice) schedDevice.addEventListener('change', () => updateToolScheduling('control_workspace_device', schedDevice.value));

/**
 * UI Utilities
 */
function updateStatus(text, cls) {
  if (!statusBadge) return;
  statusBadge.textContent = text;
  statusBadge.className = `badge badge-${cls}`;
}

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Button Events
if (connectBtn) connectBtn.addEventListener('click', connect);
if (disconnectBtn) disconnectBtn.addEventListener('click', disconnect);

if (muteBtn) {
  muteBtn.addEventListener('click', () => {
    isMuted = !isMuted;
    muteBtn.textContent = isMuted ? '🔇 Unmute Mic' : '🎤 Mute Mic';
    muteBtn.className = isMuted ? 'btn btn-warning' : 'btn btn-secondary';
  });
}

if (interruptBtn) {
  interruptBtn.addEventListener('click', () => {
    handleInterruption();
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'client_content',
        text: 'Hold on a second, Aria.',
        turn_complete: true
      }));
    }
  });
}

// Toggle Thoughts
window.toggleThoughts = function() {
  const el = document.getElementById('thoughtContainer');
  const btn = document.getElementById('toggleThoughtBtn');
  if (el.style.display === 'none') {
    el.style.display = 'block';
    btn.textContent = 'Hide Reasoning';
  } else {
    el.style.display = 'none';
    btn.textContent = 'Show Reasoning';
  }
};
