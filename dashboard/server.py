"""FastAPI + WebSocket server for facial-hci-agent dashboard."""
from __future__ import annotations
import base64
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.facial_hci.agent import FacialHCIAgent
from src.facial_hci.training.personalize import PersonalProfile
from src.facial_hci.privacy import CONSENT, redact_frame_for_log
from src.facial_hci.config import settings
from src.facial_hci.logging_utils import get_logger
from src.facial_hci.perception.face_mesh import ensure_model

log = get_logger(__name__)


# ─── Lifecycle ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm MediaPipe model download
    ensure_model()
    log.info("MediaPipe model ready.")
    yield
    log.info("Shutting down...")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
templates = Jinja2Templates(directory="dashboard/templates")


# ─── Session registry ─────────────────────────────────────────────────
class SessionRegistry:
    def __init__(self):
        self._sessions: Dict[str, FacialHCIAgent] = {}

    def get_or_create(self, session_id: str) -> FacialHCIAgent:
        if session_id not in self._sessions:
            profile = PersonalProfile.load_or_new(session_id)
            self._sessions[session_id] = FacialHCIAgent(
                session_id=session_id,
                personalization=profile,
            )
        return self._sessions[session_id]

    def remove(self, session_id: str):
        self._sessions.pop(session_id, None)


SESSIONS = SessionRegistry()


# ─── Routes ───────────────────────────────────────────────────────────
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "llm_enabled": settings.enable_llm_reasoning,
        "llm_provider": settings.llm_provider,
    })


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True}


class ConsentRequest(BaseModel):
    user_agent: str = ""


@app.post("/api/consent")
async def grant_consent(req: ConsentRequest):
    rec = CONSENT.grant(user_agent=req.user_agent)
    return {"session_id": rec.session_id, "consented_at": rec.consented_at}


@app.post("/api/consent/revoke")
async def revoke_consent(req: ConsentRequest):
    # In real app, you'd pass session_id to revoke
    CONSENT.revoke("")  # simplistic
    return {"revoked": True}


# ─── WebSocket ───────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str = None):
    await websocket.accept()
    if not session_id:
        session_id = str(uuid.uuid4())
    agent = SESSIONS.get_or_create(session_id)
    frame_count = 0
    start_t = time.time()

    log.info(f"WebSocket connected: session={session_id}")

    try:
        while True:
            # Expect: {"frame": "base64_jpeg", "timestamp": ms}
            data = await websocket.receive_json()
            b64 = data.get("frame", "")
            if not b64:
                continue

            # Decode JPEG
            jpeg_bytes = base64.b64decode(b64)
            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Analyze
            result = agent.analyze(frame)
            frame_count += 1

            # Send back
            await websocket.send_json({
                "face_detected": result.face_detected,
                "emotion": result.emotion,
                "emotion_confidence": result.emotion_confidence,
                "emotion_probs": result.emotion_probs,
                "cognitive_state": result.cognitive_state,
                "cognitive_confidence": result.cognitive_confidence,
                "action_units": result.action_units,
                "head_pose": result.head_pose,
                "gaze": result.gaze,
                "blink": result.blink,
                "blink_rate_per_min": result.blink_rate_per_min,
                "micro_events": result.micro_events,
                "thought_inference": result.thought_inference,
                "fps": frame_count / (time.time() - start_t + 1e-6),
            })

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected: session={session_id}")
    finally:
        SESSIONS.remove(session_id)


# ─── Run entrypoint ───────────────────────────────────────────────────
def run():
    import uvicorn
    uvicorn.run("dashboard.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
