# 🧠 Facial HCI Agent v1.0

**Research-grounded real-time facial analysis for Human-Computer Interaction.**

Built on the **Facial Action Coding System (FACS)** — Ekman & Friesen (1978) — with
LLM reasoning (Sălăgean et al. 2025), per-user personalization, and a live
browser dashboard. Production-ready with comprehensive error handling, rate limiting,
and session management.

---

## ✨ Features

### Core Capabilities
- 🎯 **20 FACS Action Units** extracted from MediaPipe blendshapes
- 🎭 **7 emotions** (Ekman): happiness, sadness, fear, surprise, disgust, anger, contempt
- 🧩 **7 cognitive states**: engaged, attentive, high_cognitive_load, stressed, confused, bored, distracted
- 👁 **Iris-based gaze estimation**, **head pose**, **blink rate**
- ⚡ **Micro-expression detection** (temporal AU-spike model, < 500 ms events)
- 🤖 **LLM reasoning layer** (Groq Llama-3.3-70B free tier, or local Ollama)
- 👤 **Per-user personalization**: baseline adaptation + optional LogisticRegression classifier trained on your own tagged samples
- 🌐 **Live web dashboard**: real-time emotion timeline, AU bars, thought-inference feed
- 🔒 **Privacy-first**: explicit consent gate, no raw video stored, GDPR-aware logs
- 🚀 **Deploys to Railway** on the free tier

### Production Features (v1.0)
- **Rate Limiting**: 15 FPS per session to prevent abuse
- **Session Management**: Automatic cleanup of stale sessions (5 min timeout)
- **Error Handling**: Comprehensive try-catch blocks with graceful degradation
- **CORS Support**: Configured for cross-origin requests
- **Health Checks**: Detailed health endpoint with system status
- **Consent TTL**: Consent records expire after 24 hours
- **Input Validation**: Pydantic validators for all inputs
- **Logging**: Structured logging with configurable levels
- **WebSocket Timeout**: 30-second timeout for inactive connections
- **Graceful Shutdown**: Proper cleanup of resources on shutdown

---

## 🏗 Architecture



Browser webcam (getUserMedia)
│ WebSocket — JPEG frames @ 8-10 FPS
▼
FastAPI (Render)
│
├── MediaPipe FaceLandmarker  → 468 landmarks + 52 blendshapes
├── FACS AU extraction + amplification (20 AUs)
├── Head pose / gaze / blink / micro-expression buffer
├── Rule-based emotion + cognitive-state inference
├── Multimodal fusion (face + optional prosody)
├── LLM reasoner (Groq / Ollama) — FACS-grounded prompt
└── Per-user PersonalProfile (baseline + classifier)
│
▼
WebSocket → Dashboard (Chart.js live charts + feeds)



See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/RESEARCH.md`](docs/RESEARCH.md).

---

## 🚀 Quick Start

### Local dev

```bash
# 1. Clone & install
git clone https://github.com/mbhat24/facial-hci-agent.git
cd facial-hci-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Get a FREE Groq API key at https://console.groq.com/keys
cp .env.example .env
# edit .env → paste GROQ_API_KEY=gsk_xxx

# 3. Run the web dashboard
python scripts/run_dashboard.py
# → open http://localhost:8000
```

**CLI (webcam in OpenCV window — great for dev)**
```bash
python scripts/run_agent.py
```

**Collect personal training samples**
```bash
python scripts/collect_training_data.py my_username
# Press n/h/s/f/c/a to tag moments, t to train, q to quit
```

**Run tests**
```bash
pytest tests/ -v
```

---

## ☁️ Deploy to Railway (recommended - free, works with MediaPipe)

Railway's free tier includes the OpenGL libraries required for MediaPipe.

**Steps:**
1. Push this repo to GitHub.
2. Go to https://railway.app/ → New Project → Deploy from GitHub repo.
3. Connect your GitHub and select `facial-hci-agent`.
4. Click "Deploy now".
5. After deployment, go to the project → Variables tab.
6. Add environment variable: `GROQ_API_KEY` (from https://console.groq.com/keys).
7. Click "Save" and Railway will redeploy.
8. Your dashboard is live at `https://<your-app>.up.railway.app`

**Notes for Railway free tier**
- First request after idle (cold-start) takes ~20-30 seconds.
- WebSockets are fully supported.
- Frame rate is rate-limited to 15 FPS per session.
- The MediaPipe model is downloaded during build.

---

## ☁️ Alternative: Render (Docker deployment)

Render's free tier requires Docker deployment to include OpenGL libraries.

**Steps:**
1. Delete any existing Render service for this repo.
2. Go to https://render.com/ → New → Blueprint.
3. Connect GitHub and select `facial-hci-agent`.
4. Render will read `render.yaml` (configured for Docker).
5. Set `GROQ_API_KEY` environment variable in the dashboard.
6. Deploy. Your dashboard is live at `https://<your-app>.onrender.com`

**Notes for Render**
- Uses Docker deployment with OpenGL libraries pre-installed.
- Same cold-start behavior as Railway.

---

## 🎛 Configuration (.env)

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | — | Free key from console.groq.com |
| `LLM_PROVIDER` | groq | groq / ollama / none |
| `LLM_MODEL` | llama-3.3-70b-versatile | Groq model or local Ollama tag |
| `OLLAMA_HOST` | http://localhost:11434 | If using local Ollama |
| `ENABLE_LLM_REASONING` | true | Disable to run rules-only |
| `LLM_COOLDOWN_SECONDS` | 2.5 | Min gap between LLM calls |
| `LOG_LEVEL` | INFO | DEBUG / INFO / WARNING |

---

## 🔒 Ethics & Limitations

**Please read `ETHICS.md` before deploying.**

- Facial inference is probabilistic, not truth. A smile ≠ happy (Barrett et al. 2019).
- Accuracy varies by skin tone, age, gender, culture. Known bias exists.
- Never use outputs for hiring, medical, legal, or policing decisions.
- GDPR Art. 9: facial biometrics are "special category" data. Get explicit consent.
- This repo gates the camera behind a consent modal — keep it.

---

## 📚 Research Foundations

- Ekman & Friesen (1978) — Facial Action Coding System
- Sălăgean, Leba, Ionica (2025) — Micro-expression recognition with AUs + GPT (93.3% on CASME II). [Paper](https://www.mdpi.com/2076-3417/15/12/6417)
- Barrett et al. (2019) — Emotional Expressions Reconsidered (critical view)
- Baltrusaitis et al. (2018) — OpenFace 2.0
- Cheong et al. (2023) — Py-Feat

Full citations in [`docs/RESEARCH.md`](docs/RESEARCH.md).

---

## 🧪 Project layout

```
facial-hci-agent/
├── src/facial_hci/
│   ├── perception/    (face_mesh, head_pose, iris_gaze)
│   ├── features/      (action_units, blink, micro-expressions, prosody)
│   ├── inference/     (emotion_rules, cognitive_state, fusion, llm_reasoner)
│   ├── training/      (personalize, evaluate)
│   ├── agent.py       ← orchestrator
│   ├── privacy.py
│   └── config.py
├── dashboard/
│   ├── server.py      ← FastAPI + WebSocket
│   ├── templates/index.html
│   └── static/css,js
├── scripts/           (run_agent, run_dashboard, collect_training_data)
├── tests/             (pytest unit tests)
├── docs/              (ARCHITECTURE, RESEARCH)
├── render.yaml
├── Procfile
└── requirements.txt
```

---

## 📄 License

MIT — see [`LICENSE`](LICENSE).

---

## 🙏 Contributing

PRs welcome. Please add a test for any behavior change and keep
the consent gate intact.
