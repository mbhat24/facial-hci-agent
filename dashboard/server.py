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
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.facial_hci.agent import FacialHCIAgent
from src.facial_hci.training.personalize import PersonalProfile
from src.facial_hci.privacy import CONSENT, redact_frame_for_log
from src.facial_hci.config import settings
from src.facial_hci.logging_utils import get_logger
from src.facial_hci.perception.face_mesh import ensure_model
import redis

log = get_logger(__name__)

# Initialize Redis client if enabled
redis_client = None
if settings.enable_redis:
    try:
        redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        redis_client.ping()
        log.info(f"Redis connected: {settings.redis_url}")
    except Exception as e:
        log.warning(f"Redis connection failed, falling back to in-memory: {e}")
        redis_client = None

# Rate limiting: max frames per second per session
MAX_FPS_PER_SESSION = 15
FRAME_INTERVAL = 1.0 / MAX_FPS_PER_SESSION
SESSION_TIMEOUT = 300  # 5 minutes
HEARTBEAT_INTERVAL = 30  # Send ping every 30 seconds
HEARTBEAT_TIMEOUT = 35  # Disconnect if no pong after 35 seconds

# Prometheus metrics
frames_processed_total = Counter('frames_processed_total', 'Total frames processed', ['session_id'])
frame_processing_duration = Histogram('frame_processing_duration_seconds', 'Frame processing duration')
active_sessions = Gauge('active_sessions', 'Number of active sessions')
llm_calls_total = Counter('llm_calls_total', 'Total LLM API calls', ['status'])
llm_call_duration = Histogram('llm_call_duration_seconds', 'LLM call duration')
websocket_connections_total = Counter('websocket_connections_total', 'Total WebSocket connections', ['status'])
session_duration = Histogram('session_duration_seconds', 'Session duration')


# ─── Lifecycle ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm MediaPipe model download
    ensure_model()
    log.info("MediaPipe model ready.")
    
    # Initialize Prometheus instrumentator
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
    log.info("Prometheus metrics enabled at /metrics")
    
    yield
    log.info("Shutting down...")


app = FastAPI(
    lifespan=lifespan,
    title="Facial HCI Agent",
    description="""Research-grounded real-time facial analysis for Human-Computer Interaction.

## Features
- 20 FACS Action Units extracted from MediaPipe blendshapes
- 7 emotions (Ekman): happiness, sadness, fear, surprise, disgust, anger, contempt
- 7 cognitive states: engaged, attentive, high_cognitive_load, stressed, confused, bored, distracted
- Iris-based gaze estimation, head pose, blink rate
- Micro-expression detection (temporal AU-spike model, < 500 ms events)
- LLM reasoning layer (Groq Llama-3.3-70B free tier, or local Ollama)

## Privacy
- Explicit consent gate
- No raw video stored
- GDPR-compliant data export/delete endpoints
""",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "health",
            "description": "Health check and monitoring endpoints"
        },
        {
            "name": "consent",
            "description": "User consent management"
        },
        {
            "name": "data",
            "description": "GDPR data export and deletion"
        },
        {
            "name": "websocket",
            "description": "Real-time WebSocket video analysis"
        }
    ]
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
        self._session_start: Dict[str, float] = {}
        self._use_redis = redis_client is not None

    def _set_redis(self, key: str, value: str, ttl: int = SESSION_TIMEOUT):
        """Set value in Redis with TTL if enabled."""
        if self._use_redis:
            redis_client.setex(f"session:{key}", ttl, value)

    def _get_redis(self, key: str) -> Optional[str]:
        """Get value from Redis if enabled."""
        if self._use_redis:
            return redis_client.get(f"session:{key}")
        return None

    def _delete_redis(self, key: str):
        """Delete value from Redis if enabled."""
        if self._use_redis:
            redis_client.delete(f"session:{key}")

    def get_or_create(self, session_id: str) -> FacialHCIAgent:
        # Check if session exists in Redis first
        if self._use_redis and self._get_redis(session_id):
            if session_id not in self._sessions:
                # Session exists in Redis but not in memory - recreate agent
                profile = PersonalProfile.load_or_new(session_id)
                self._sessions[session_id] = FacialHCIAgent(
                    session_id=session_id,
                    personalization=profile,
                )
        elif session_id not in self._sessions:
            profile = PersonalProfile.load_or_new(session_id)
            self._sessions[session_id] = FacialHCIAgent(
                session_id=session_id,
                personalization=profile,
            )
            self._session_start[session_id] = time.time()
            active_sessions.inc()
            # Store in Redis
            self._set_redis(session_id, "active")
        
        self._last_activity[session_id] = time.time()
        # Update Redis TTL
        if self._use_redis:
            redis_client.expire(f"session:{session_id}", SESSION_TIMEOUT)
        
        return self._sessions[session_id]

    def remove(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id].close()
            del self._sessions[session_id]
        self._last_activity.pop(session_id, None)
        self._frame_count.pop(session_id, None)
        
        # Track session duration
        if session_id in self._session_start:
            duration = time.time() - self._session_start[session_id]
            session_duration.observe(duration)
            del self._session_start[session_id]
        
        # Remove from Redis
        self._delete_redis(session_id)
        
        active_sessions.dec()

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


@app.get("/health", tags=["health"])
async def health():
    """
    Health check endpoint with system status.
    
    Returns:
        JSON with service status, active sessions, and configuration
    """
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
            "version": "1.1.0",
            "redis_enabled": redis_client is not None
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


@app.post("/api/consent", tags=["consent"])
async def grant_consent(req: ConsentRequest):
    """
    Grant consent for facial analysis.
    
    Args:
        req: Consent request with user agent
        
    Returns:
        Session ID and consent timestamp
    """
    try:
        rec = CONSENT.grant(user_agent=req.user_agent)
        log.info(f"Consent granted for session: {rec.session_id}")
        return {"session_id": rec.session_id, "consented_at": rec.consented_at}
    except Exception as e:
        log.error(f"Error granting consent: {e}")
        raise HTTPException(status_code=500, detail="Failed to grant consent")


@app.post("/api/consent/revoke", tags=["consent"])
async def revoke_consent(req: ConsentRequest):
    """
    Revoke consent for facial analysis.
    
    Args:
        req: Consent request
        
    Returns:
        Revocation confirmation
    """
    try:
        CONSENT.revoke("")  # In production, pass session_id
        log.info("Consent revoked")
        return {"revoked": True}
    except Exception as e:
        log.error(f"Error revoking consent: {e}")
        raise HTTPException(status_code=500, detail="Failed to revoke consent")


class DataExportRequest(BaseModel):
    session_id: str


class DataDeleteRequest(BaseModel):
    session_id: str


@app.post("/api/data/export", tags=["data"])
async def export_data(req: DataExportRequest):
    """
    Export all data for a session (GDPR right to data portability).
    
    Args:
        req: Data export request with session ID
        
    Returns:
        Session data including consent, profile, and activity metrics
    """
    try:
        # Check if session exists
        if req.session_id not in SESSIONS._sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Collect session data
        session_data = {
            "session_id": req.session_id,
            "session_start": SESSIONS._session_start.get(req.session_id),
            "last_activity": SESSIONS._last_activity.get(req.session_id),
            "frame_count": SESSIONS._frame_count.get(req.session_id, 0),
            "consent_record": None,
            "profile_data": None,
        }
        
        # Get consent record if available
        if CONSENT.is_valid(req.session_id):
            # In production, retrieve full consent record
            session_data["consent_record"] = "valid"
        
        # Get profile data if available
        profile = PersonalProfile.load_or_new(req.session_id)
        if profile:
            session_data["profile_data"] = {
                "has_baseline": profile.baseline_au is not None,
                "has_classifier": profile.clf is not None,
            }
        
        log.info(f"Data export for session: {req.session_id}")
        return session_data
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error exporting data for session {req.session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to export data")


@app.post("/api/data/delete", tags=["data"])
async def delete_data(req: DataDeleteRequest):
    """
    Delete all data for a session (GDPR right to be forgotten).
    
    Args:
        req: Data deletion request with session ID
        
    Returns:
        Deletion confirmation
    """
    try:
        # Remove session
        SESSIONS.remove(req.session_id)
        
        # Revoke consent
        CONSENT.revoke(req.session_id)
        
        # Delete profile data
        profile_path = settings.profiles_dir / f"{req.session_id}.json"
        if profile_path.exists():
            profile_path.unlink()
        
        # Delete log files
        log_path = settings.data_dir / f"{req.session_id}.jsonl"
        if log_path.exists():
            log_path.unlink()
        
        log.info(f"Data deleted for session: {req.session_id}")
        return {"deleted": True, "session_id": req.session_id}
        
    except Exception as e:
        log.error(f"Error deleting data for session {req.session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete data")


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
    last_heartbeat = time.time()

    log.info(f"WebSocket connected: session={session_id}")
    websocket_connections_total.labels(status='connected').inc()

    async def heartbeat_loop():
        """Send periodic pings to detect dead connections."""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            try:
                await websocket.send_json({"type": "ping"})
                log.debug(f"Heartbeat sent for session {session_id}")
            except:
                log.warning(f"Heartbeat failed for session {session_id}")
                break

    heartbeat_task = asyncio.create_task(heartbeat_loop())

    try:
        while True:
            # Rate limiting
            now = time.time()
            if now - last_frame_time < FRAME_INTERVAL:
                await asyncio.sleep(FRAME_INTERVAL - (now - last_frame_time))
            last_frame_time = now

            # Check heartbeat timeout
            if now - last_heartbeat > HEARTBEAT_TIMEOUT:
                log.warning(f"Heartbeat timeout for session {session_id}")
                break

            # Expect: {"frame": "base64_jpeg", "timestamp": ms} or {"type": "pong"}
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30)
            except asyncio.TimeoutError:
                log.warning(f"WebSocket timeout for session {session_id}")
                websocket_connections_total.labels(status='timeout').inc()
                break
            
            # Handle heartbeat response
            if data.get("type") == "pong":
                last_heartbeat = time.time()
                continue
            
            b64 = data.get("frame", "")
            if not b64:
                continue

            # Decode WebP (better compression than JPEG)
            try:
                webp_bytes = base64.b64decode(b64)
                frame = cv2.imdecode(np.frombuffer(webp_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception as e:
                log.error(f"Frame decode error for session {session_id}: {e}")
                continue

            # Analyze with metrics
            try:
                with frame_processing_duration.time():
                    result = agent.analyze(frame)
                frame_count += 1
                SESSIONS._frame_count[session_id] = frame_count
                SESSIONS._last_activity[session_id] = time.time()
                frames_processed_total.labels(session_id=session_id).inc()
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
        websocket_connections_total.labels(status='disconnected').inc()
    except Exception as e:
        log.error(f"WebSocket error for session {session_id}: {e}")
        websocket_connections_total.labels(status='error').inc()
    finally:
        heartbeat_task.cancel()
        SESSIONS.remove(session_id)


# ─── Run entrypoint ───────────────────────────────────────────────────
def run():
    import uvicorn
    uvicorn.run("dashboard.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
