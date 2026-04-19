"""Consent gating and data-minimization helpers."""
from dataclasses import dataclass, field
from typing import Dict
import time
import uuid


@dataclass
class ConsentRecord:
    session_id: str
    consented_at: float
    user_agent: str = ""
    note: str = ""


class ConsentRegistry:
    """In-memory consent registry. For production, move to a DB."""

    def __init__(self):
        self._records: Dict[str, ConsentRecord] = {}

    def grant(self, user_agent: str = "", note: str = "") -> ConsentRecord:
        sid = uuid.uuid4().hex
        rec = ConsentRecord(session_id=sid, consented_at=time.time(),
                            user_agent=user_agent, note=note)
        self._records[sid] = rec
        return rec

    def is_valid(self, sid: str) -> bool:
        return sid in self._records

    def revoke(self, sid: str):
        self._records.pop(sid, None)


CONSENT = ConsentRegistry()


def redact_frame_for_log(frame_meta: dict) -> dict:
    """Remove anything that could be used to reidentify a user from logs."""
    banned = {"raw_frame", "jpeg", "image", "pixels"}
    return {k: v for k, v in frame_meta.items() if k not in banned}
