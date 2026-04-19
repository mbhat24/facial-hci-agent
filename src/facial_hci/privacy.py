"""Consent gating and data-minimization helpers."""
from dataclasses import dataclass, field
from typing import Dict, Optional
import time
import uuid
import logging

log = logging.getLogger(__name__)

CONSENT_TTL = 86400  # 24 hours


@dataclass
class ConsentRecord:
    session_id: str
    consented_at: float
    user_agent: str = ""
    note: str = ""
    revoked_at: Optional[float] = None


class ConsentRegistry:
    """In-memory consent registry with TTL. For production, use a database."""

    def __init__(self, ttl: int = CONSENT_TTL):
        self._records: Dict[str, ConsentRecord] = {}
        self._ttl = ttl

    def grant(self, user_agent: str = "", note: str = "") -> ConsentRecord:
        sid = uuid.uuid4().hex
        rec = ConsentRecord(
            session_id=sid,
            consented_at=time.time(),
            user_agent=user_agent[:500],  # Truncate for safety
            note=note[:200]
        )
        self._records[sid] = rec
        log.info(f"Consent granted for session {sid}")
        return rec

    def is_valid(self, sid: str) -> bool:
        if sid not in self._records:
            return False
        rec = self._records[sid]
        if rec.revoked_at is not None:
            return False
        if time.time() - rec.consented_at > self._ttl:
            self._records.pop(sid, None)
            return False
        return True

    def revoke(self, sid: str):
        if sid in self._records:
            self._records[sid].revoked_at = time.time()
            log.info(f"Consent revoked for session {sid}")

    def cleanup_expired(self) -> int:
        """Remove expired consent records. Returns count removed."""
        now = time.time()
        expired = [
            sid for sid, rec in self._records.items()
            if (rec.revoked_at is not None and now - rec.revoked_at > self._ttl) or
               (rec.revoked_at is None and now - rec.consented_at > self._ttl)
        ]
        for sid in expired:
            del self._records[sid]
        if expired:
            log.info(f"Cleaned up {len(expired)} expired consent records")
        return len(expired)


CONSENT = ConsentRegistry()


def redact_frame_for_log(frame_meta: dict) -> dict:
    """Remove anything that could be used to reidentify a user from logs."""
    banned = {"raw_frame", "jpeg", "image", "pixels"}
    return {k: v for k, v in frame_meta.items() if k not in banned}
