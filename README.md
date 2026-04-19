# 🧠 Facial HCI Agent

**Research-grounded real-time facial analysis for Human-Computer Interaction.**

Built on the **Facial Action Coding System (FACS)** — Ekman & Friesen (1978) — with
LLM reasoning (Sălăgean et al. 2025), per-user personalization, and a live
browser dashboard. Deploys to Render on the free tier.

---

## ✨ Features

- 🎯 **20 FACS Action Units** extracted from MediaPipe blendshapes
- 🎭 **7 emotions** (Ekman): happiness, sadness, fear, surprise, disgust, anger, contempt
- 🧩 **7 cognitive states**: engaged, attentive, high_cognitive_load, stressed, confused, bored, distracted
- 👁 **Iris-based gaze estimation**, **head pose**, **blink rate**
- ⚡ **Micro-expression detection** (temporal AU-spike model, < 500 ms events)
- 🤖 **LLM reasoning layer** (Groq Llama-3.3-70B free tier, or local Ollama)
- 👤 **Per-user personalization**: baseline adaptation + optional LogisticRegression classifier trained on your own tagged samples
- 🌐 **Live web dashboard**: real-time emotion timeline, AU bars, thought-inference feed
- 🔒 **Privacy-first**: explicit consent gate, no raw video stored, GDPR-aware logs
- 🚀 **Deploys to Render** in one click via `render.yaml`

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

## ☁️ Deploy to Render (free, 5 min)

**Option A — One-click via render.yaml**
1. Push this repo to GitHub.
2. Go to https://render.com/ → New → Blueprint.
3. Connect your GitHub and pick this repo. Render detects `render.yaml`.
4. In the Render dashboard, click into the service → Environment → set `GROQ_API_KEY` (from https://console.groq.com/keys).
5. Deploy. Your dashboard is live at `https://<your-app>.onrender.com`.

**Option B — Manual**
- Service type: Web Service · Runtime: Python 3.11 · Plan: Free
- Build: `pip install -r requirements.txt`
- Start: `uvicorn dashboard.server:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`
- Env var: `GROQ_API_KEY`

**Notes for Render free tier**
- First request after idle (15 min) will cold-start (20 s).
- WebSockets are fully supported.
- Keep frame rate ≤ 10 FPS to stay within CPU budget.
- The MediaPipe model (~3 MB) is downloaded at build time.

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
