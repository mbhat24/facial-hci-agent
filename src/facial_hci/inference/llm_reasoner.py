"""LLM reasoning layer — Groq (cloud, free) or Ollama (local).

Implements the GPT-reasoning approach from Sălăgean et al. (2025):
  AU evidence + context  ->  single-sentence mental-state hypothesis.
"""
from __future__ import annotations
import json
import time
from typing import Dict, List, Optional

import httpx
from ..config import settings
from ..logging_utils import get_logger

log = get_logger(__name__)


SYSTEM_PROMPT = """You are an affective-computing expert using the Facial Action Coding System (FACS, Ekman & Friesen).
Given observed Action Units, head pose, gaze, and cognitive heuristics, infer the user's most LIKELY mental state in ONE short sentence (< 20 words).

Rules:
- You are NOT reading thoughts. Say "possibly" / "may be" / "appears" — never assert certainty.
- Ground your answer in the AU evidence provided.
- Respond in this format exactly:
  STATE: <one-sentence interpretation>
- Do not output anything else.

FACS quick reference:
  AU1+AU2  = brows raised (surprise/interest)
  AU4      = brow furrow (concentration OR negative affect)
  AU5      = upper-lid raise (alertness/fear)
  AU6+AU12 = genuine (Duchenne) smile
  AU12     = lip-corner pull (polite smile)
  AU9+AU10 = disgust
  AU15+AU1+AU17 = sadness
  AU20     = lip stretch (fear)
  AU23     = lip tightening (suppression/anger)
"""


def _build_user_prompt(
    active_aus: Dict[str, float],
    head_pose: Dict[str, float],
    gaze: Dict[str, float],
    emotion: str,
    cognitive: str,
    history: List[str],
    micro_events: List[str],
) -> str:
    return (
        f"Active AUs (0-1): {json.dumps(active_aus)}\n"
        f"Head pose deg: {json.dumps({k: round(v, 1) for k, v in head_pose.items()})}\n"
        f"Gaze: {json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in gaze.items()})}\n"
        f"Rule-based emotion: {emotion}\n"
        f"Rule-based cognitive state: {cognitive}\n"
        f"Recent micro-expression events: {micro_events[-3:] if micro_events else []}\n"
        f"Recent history: {history[-3:] if history else []}\n\n"
        f"Give ONE sentence starting with STATE:"
    )


class LLMReasoner:
    """Unified interface — Groq or Ollama."""

    def __init__(self):
        self.provider = settings.llm_provider.lower()
        self.model = settings.llm_model
        self.cooldown = settings.llm_cooldown_seconds
        self._last_call = 0.0
        self._client = None

        if not settings.enable_llm_reasoning or self.provider == "none":
            log.info("LLM reasoning disabled.")
            self.provider = "none"
            return

        if self.provider == "groq":
            if not settings.groq_api_key:
                log.warning("GROQ_API_KEY missing — disabling LLM.")
                self.provider = "none"
                return
            try:
                from groq import Groq
                self._client = Groq(api_key=settings.groq_api_key)
                log.info(f"Groq reasoner ready ({self.model}).")
            except Exception as e:
                log.error(f"Groq init failed: {e}")
                self.provider = "none"
        elif self.provider == "ollama":
            log.info(f"Ollama reasoner ready ({self.model} @ {settings.ollama_host}).")
        else:
            log.warning(f"Unknown LLM provider: {self.provider}")
            self.provider = "none"

    def available(self) -> bool:
        return self.provider != "none"

    def reason(
        self,
        active_aus: Dict[str, float],
        head_pose: Dict[str, float],
        gaze: Dict[str, float],
        emotion: str,
        cognitive: str,
        history: List[str],
        micro_events: Optional[List[str]] = None,
    ) -> str:
        if not self.available():
            return ""
        now = time.time()
        if now - self._last_call < self.cooldown:
            return ""
        self._last_call = now

        user = _build_user_prompt(active_aus, head_pose, gaze, emotion,
                                  cognitive, history, micro_events or [])

        try:
            if self.provider == "groq":
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=70,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()
            elif self.provider == "ollama":
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 70},
                }
                r = httpx.post(f"{settings.ollama_host}/api/chat",
                               json=payload, timeout=20.0)
                r.raise_for_status()
                return r.json()["message"]["content"].strip()
        except Exception as e:
            log.error(f"LLM error: {e}")
            return ""
        return ""
