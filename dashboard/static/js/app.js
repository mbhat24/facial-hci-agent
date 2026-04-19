// ─── State ───────────────────────────────────────────────────────────
const state = {
  ws: null,
  video: null,
  canvas: null,
  ctx: null,
  running: false,
  sessionId: null,
  history: [],
  maxHistory: 100,
  frameInterval: 100,  // ms
  lastFrame: 0,
  adaptiveFrameInterval: 1000 / 10,  // Adaptive frame rate
  networkLatency: 0,  // Track network latency
  rttHistory: [],  // Round-trip time history
  maxRttHistory: 10,
};

const $ = id => document.getElementById(id);

// ─── Consent ─────────────────────────────────────────────────────────
async function initConsent() {
  const res = await fetch("/api/consent", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({user_agent: navigator.userAgent}),
  });
  const data = await res.json();
  state.sessionId = data.session_id;
}

$("btnAgree").onclick = async () => {
  console.log("Agree clicked, initializing consent...");
  await initConsent();
  console.log("Consent initialized, session_id:", state.sessionId);
  $("consentModal").style.display = "none";
  $("btnStart").disabled = false;
  console.log("Modal hidden, Start button enabled");
};

$("btnDecline").onclick = () => {
  window.location.href = "about:blank";
};

// ─── Camera ─────────────────────────────────────────────────────────
async function startCamera() {
  try {
    console.log("Starting camera...");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    console.log("Camera stream obtained");
    state.video = $("video");
    state.video.srcObject = stream;
    await state.video.play();
    console.log("Video playing");

    state.canvas = $("overlay");
    state.ctx = state.canvas.getContext("2d");
    state.canvas.width = 640;
    state.canvas.height = 480;

    $("btnStart").disabled = true;
    $("btnStop").disabled = false;
    $("btnReset").disabled = false;
    state.running = true;
    console.log("Connecting WebSocket...");
    connectWebSocket();
    console.log("Starting frame loop...");
    requestAnimationFrame(frameLoop);
  } catch (err) {
    console.error("Camera error:", err);
    alert("Camera error: " + err.message);
  }
}

function stopCamera() {
  state.running = false;
  if (state.video && state.video.srcObject) {
    state.video.srcObject.getTracks().forEach(t => t.stop());
    state.video.srcObject = null;
  }
  if (state.ws) {
    state.ws.close();
    state.ws = null;
  }
  $("btnStart").disabled = false;
  $("btnStop").disabled = true;
}

$("btnStart").onclick = startCamera;
$("btnStop").onclick = stopCamera;

// ─── WebSocket ──────────────────────────────────────────────────────
function connectWebSocket() {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${proto}//${window.location.host}/ws?session_id=${state.sessionId}`;
  state.ws = new WebSocket(wsUrl);

  state.ws.onopen = () => {
    console.log("WebSocket connected with session:", state.sessionId);
  };

  state.ws.onmessage = (event) => {
    const d = JSON.parse(event.data);
    const receiveTime = Date.now();
    
    // Handle heartbeat ping
    if (d.type === "ping") {
      state.ws.send(JSON.stringify({type: "pong"}));
      console.log("Pong sent");
      return;
    }
    
    // Calculate RTT and update adaptive frame rate
    if (state.lastSendTime) {
      const rtt = receiveTime - state.lastSendTime;
      updateAdaptiveFrameRate(rtt);
    }
    
    console.log("WebSocket message received");
    console.log("Analysis data:", d);
    renderAnalysis(d);
  };

  state.ws.onclose = () => {
    if (state.running) {
      console.log("WebSocket closed, reconnecting in 2s...");
      setTimeout(connectWebSocket, 2000);
    }
  };

  state.ws.onerror = (err) => {
    console.error("WebSocket error:", err);
  };
}

// ─── Frame loop ─────────────────────────────────────────────────────
function frameLoop(timestamp) {
  if (!state.running) return;

  if (timestamp - state.lastFrame >= state.adaptiveFrameInterval) {
    state.lastFrame = timestamp;

    // Downsample to 320x240 for bandwidth
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 320;
    tempCanvas.height = 240;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(state.video, 0, 0, 320, 240);

    // Send as JPEG with RTT tracking
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      const sendTime = Date.now();
      const jpeg = tempCanvas.toDataURL("image/jpeg", 0.7);
      const b64 = jpeg.split(",")[1];
      
      state.ws.send(JSON.stringify({
        frame: b64,
        timestamp: sendTime,
      }));
      
      // Track send time for RTT calculation
      state.lastSendTime = sendTime;
    } else {
      console.log("WebSocket not ready, state:", state.ws ? state.ws.readyState : "null");
    }
  }

  requestAnimationFrame(frameLoop);
}

// ─── Adaptive frame rate ─────────────────────────────────────────────
function updateAdaptiveFrameRate(rtt) {
  // Update RTT history
  state.rttHistory.push(rtt);
  if (state.rttHistory.length > state.maxRttHistory) {
    state.rttHistory.shift();
  }
  
  // Calculate average RTT
  const avgRtt = state.rttHistory.reduce((a, b) => a + b, 0) / state.rttHistory.length;
  
  // Adjust frame interval based on RTT
  // Target: keep RTT below 100ms
  const targetRtt = 100;
  const minFrameInterval = 1000 / 15;  // 15 FPS max
  const maxFrameInterval = 1000 / 5;   // 5 FPS min
  
  if (avgRtt > targetRtt * 2) {
    // High latency - reduce frame rate
    state.adaptiveFrameInterval = Math.min(maxFrameInterval, state.adaptiveFrameInterval * 1.2);
  } else if (avgRtt < targetRtt * 0.5) {
    // Low latency - increase frame rate
    state.adaptiveFrameInterval = Math.max(minFrameInterval, state.adaptiveFrameInterval * 0.9);
  }
  
  console.log(`RTT: ${avgRtt.toFixed(0)}ms, Frame interval: ${state.adaptiveFrameInterval.toFixed(0)}ms`);
}

// ─── Rendering ─────────────────────────────────────────────────
function renderAnalysis(d) {
  if (!d.face_detected) {
    $("statusLine").textContent = "No face detected.";
    return;
  }
  $("statusLine").textContent = "✓ Analyzing";

  // Emotion pill
  $("emotionPill").className = `state-pill emotion-${d.emotion}`;
  $("emotionPill").textContent = `${d.emotion} (${(d.emotion_confidence*100).toFixed(0)}%)`;

  // Cognitive pill
  $("cogPill").className = `state-pill ${d.cognitive_state}`;
  $("cogPill").textContent = d.cognitive_state.replace(/_/g, " ");

  // Metrics
  $("metricBlink").textContent = Math.round(d.blink_rate_per_min);
  $("metricYaw").textContent = (d.head_pose.yaw || 0).toFixed(0) + "°";
  $("metricPitch").textContent = (d.head_pose.pitch || 0).toFixed(0) + "°";
  $("metricGaze").textContent = d.gaze && d.gaze.available
     ? `${d.gaze.gaze_x.toFixed(2)},${d.gaze.gaze_y.toFixed(2)}` : "—";

  // AU bars (top 8)
  const sorted = Object.entries(d.action_units || {})
    .sort((a,b) => b[1]-a[1]).slice(0, 8);
  $("auBars").innerHTML = sorted.map(([au,v]) => `
    <div class="au-bar">
      <div class="label">${au}</div>
      <div class="track"><div class="fill" style="width:${(v*100).toFixed(0)}%"></div></div>
      <div class="val">${v.toFixed(2)}</div>
    </div>`).join("");

  // Emotion timeline
  state.history.push({t: Date.now(), probs: d.emotion_probs || {}});
  if (state.history.length > state.maxHistory) state.history.shift();
  updateEmotionChart();

  // Micro-expressions
  if (d.micro_events && d.micro_events.length) {
    const tag = document.createElement("div");
    tag.className = "thought";
    tag.innerHTML = `<span class="time">${new Date().toLocaleTimeString()}</span>
                     µ-expr: ${d.micro_events.join(", ")}`;
    $("microFeed").prepend(tag);
    while ($("microFeed").children.length > 10) $("microFeed").lastChild.remove();
  }

  // Thought inference
  if (d.thought_inference) {
    const t = document.createElement("div");
    t.className = "thought";
    t.innerHTML = `<span class="time">${new Date().toLocaleTimeString()}</span>${d.thought_inference}`;
    $("thoughtFeed").prepend(t);
    while ($("thoughtFeed").children.length > 20) $("thoughtFeed").lastChild.remove();
  }
}

// ─── Chart.js setup ────────────────────────────────────────────
let emotionChart = null;
function initChart() {
  const labels = ["happiness","sadness","anger","fear","surprise","disgust","contempt","neutral"];
  emotionChart = new Chart($("chartEmotion").getContext("2d"), {
    type: "line",
    data: {
      labels: [],
      datasets: labels.map((l,i) => ({
        label: l, data: [], borderWidth: 1.5, tension: 0.3,
        pointRadius: 0, borderColor: colorFor(l),
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      scales: {
        x: { display: false },
        y: { min: 0, max: 1, ticks: { color: "#8b98a9", font: { size: 10 }}, grid: { color: "#263040" }},
      },
      plugins: { legend: { labels: { color: "#e6edf3", font: { size: 10 }, boxWidth: 10 }}}
    }
  });
}
function colorFor(label) {
  return {
    happiness: "#3fd17a", sadness: "#6aa6e5", anger: "#ff6b6b",
    fear: "#c874ff", surprise: "#ffb547", disgust: "#9fd86a",
    contempt: "#ff9f6a", neutral: "#8b98a9",
  }[label] || "#4ea8ff";
}
function updateEmotionChart() {
  if (!emotionChart) return;
  const labels = emotionChart.data.datasets.map(d => d.label);
  emotionChart.data.labels = state.history.map((_,i) => i);
  labels.forEach((l,i) => {
    emotionChart.data.datasets[i].data = state.history.map(h => h.probs[l] || 0);
  });
  emotionChart.update("none");
}

window.addEventListener("load", initChart);
