"""FastAPI + WebSocket server for facial-hci-agent dashboard."""
from __future__ import annotations
import base64
import json
import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Optional
from collections import defaultdict

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

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

# Rate limiting: max frames per second per session
MAX_FPS_PER_SESSION = 15
FRAME_INTERVAL = 1.0 / MAX_FPS_PER_SESSION
SESSION_TIMEOUT = 300  # 5 minutes


# ─── Lifecycle ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm MediaPipe model download
    ensure_model()
    log.info("MediaPipe model ready.")
    yield
    log.info("Shutting down...")


app = FastAPI(
    lifespan=lifespan,
    title="Facial HCI Agent",
    description="Research-grounded real-time facial analysis for Human-Computer Interaction",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")
templates = Jinja2Templates(directory="dashboard/templates")


# ─── Session registry ─────────────────────────────────────────────────
class SessionRegistry:
    def __init__(self):
        self._sessions: Dict[str, FacialHCIAgent] = {}
        self._last_activity: Dict[str, float] = {}
        self._frame_count: Dict[str, int] = defaultdict(int)

    def get_or_create(self, session_id: str) -> FacialHCIAgent:
        if session_id not in self._sessions:
            profile = PersonalProfile.load_or_new(session_id)
            self._sessions[session_id] = FacialHCIAgent(
                session_id=session_id,
                personalization=profile,
            )
            self._last_activity[session_id] = time.time()
        self._last_activity[session_id] = time.time()
        return self._sessions[session_id]

    def remove(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id].close()
            del self._sessions[session_id]
        self._last_activity.pop(session_id, None)
        self._frame_count.pop(session_id, None)

    def cleanup_stale_sessions(self):
        """Remove sessions inactive for more than SESSION_TIMEOUT."""
        now = time.time()
        stale = [sid for sid, last in self._last_activity.items() if now - last > SESSION_TIMEOUT]
        for sid in stale:
            log.info(f"Cleaning up stale session: {sid}")
            self.remove(sid)
        return len(stale)


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
    """Health check endpoint with system status."""
    try:
        # Clean up stale sessions
        stale_count = SESSIONS.cleanup_stale_sessions()
        
        return {
            "status": "ok",
            "model_loaded": True,
            "active_sessions": len(SESSIONS._sessions),
            "stale_sessions_cleaned": stale_count,
            "llm_enabled": settings.enable_llm_reasoning,
            "llm_provider": settings.llm_provider,
            "version": "1.0.0"
        }
    except Exception as e:
        log.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": str(e)}
        )


class ConsentRequest(BaseModel):
    user_agent: str = Field(default="", max_length=500)
    
    @validator('user_agent')
    def sanitize_user_agent(cls, v):
        return v[:500]  # Truncate to prevent abuse


@app.post("/api/consent")
async def grant_consent(req: ConsentRequest):
    try:
        rec = CONSENT.grant(user_agent=req.user_agent)
        log.info(f"Consent granted for session: {rec.session_id}")
        return {"session_id": rec.session_id, "consented_at": rec.consented_at}
    except Exception as e:
        log.error(f"Error granting consent: {e}")
        raise HTTPException(status_code=500, detail="Failed to grant consent")


@app.post("/api/consent/revoke")
async def revoke_consent(req: ConsentRequest):
    try:
        CONSENT.revoke("")  # In production, pass session_id
        log.info("Consent revoked")
        return {"revoked": True}
    except Exception as e:
        log.error(f"Error revoking consent: {e}")
        raise HTTPException(status_code=500, detail="Failed to revoke consent")


# ─── WebSocket ───────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str = None):
    await websocket.accept()
    if not session_id:
        session_id = str(uuid.uuid4())
    agent = SESSIONS.get_or_create(session_id)
    frame_count = 0
    start_t = time.time()
    last_frame_time = 0

    log.info(f"WebSocket connected: session={session_id}")

    try:
        while True:
            # Rate limiting
            now = time.time()
            if now - last_frame_time < FRAME_INTERVAL:
                await asyncio.sleep(FRAME_INTERVAL - (now - last_frame_time))
            last_frame_time = now

            # Expect: {"frame": "base64_jpeg", "timestamp": ms}
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30)
            except asyncio.TimeoutError:
                log.warning(f"WebSocket timeout for session {session_id}")
                break
            
            b64 = data.get("frame", "")
            if not b64:
                continue

            # Decode JPEG
            try:
                jpeg_bytes = base64.b64decode(b64)
                frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception as e:
                log.error(f"Frame decode error for session {session_id}: {e}")
                continue

            # Analyze
            try:
                result = agent.analyze(frame)
                frame_count += 1
                SESSIONS._frame_count[session_id] = frame_count
                SESSIONS._last_activity[session_id] = time.time()
            except Exception as e:
                log.error(f"Analysis error for session {session_id}: {e}")
                result = None

            # Send back
            if result:
                try:
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
                except Exception as e:
                    log.error(f"Send error for session {session_id}: {e}")
                    break

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected: session={session_id}")
    except Exception as e:
        log.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        SESSIONS.remove(session_id)


# ─── Run entrypoint ───────────────────────────────────────────────────
def run():
    import uvicorn
    uvicorn.run("dashboard.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
