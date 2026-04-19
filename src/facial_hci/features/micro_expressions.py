"""Micro-expression detection via temporal AU buffer.

Strategy (research-grounded):
  - Keep last N frames of AU vectors (default 30 = ~1 s at 30 FPS / ~3 s at 10 FPS).
  - Detect brief, high-amplitude spikes (onset < 500 ms) against user baseline.
  - A micro-expression = AU spike above baseline+sigma, duration < threshold.

This is a practical approximation. For CASME II-level accuracy you would swap
in Micron-BERT or a trained 3D-CNN; the interface below is compatible.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple
import numpy as np
import time


@dataclass
class MicroEvent:
    au: str
    peak: float
    duration_ms: float
    t_onset: float


class MicroExpressionDetector:
    def __init__(self, buffer_len: int = 30, spike_sigma: float = 2.5,
                 max_duration_ms: float = 500.0):
        self.buf: Deque[Tuple[float, Dict[str, float]]] = deque(maxlen=buffer_len)
        self.spike_sigma = spike_sigma
        self.max_duration_ms = max_duration_ms
        self._baseline_mean: Dict[str, float] = {}
        self._baseline_std: Dict[str, float] = {}
        self._baseline_ready = False
        self._baseline_buffer: List[Dict[str, float]] = []
        self._baseline_size = 60  # ~6 s at 10 FPS

    def add_frame(self, aus: Dict[str, float]) -> List[MicroEvent]:
        now = time.time()
        self.buf.append((now, aus))

        # Build baseline (neutral) in first N frames
        if not self._baseline_ready:
            self._baseline_buffer.append(aus)
            if len(self._baseline_buffer) >= self._baseline_size:
                self._compute_baseline()
            return []

        return self._detect_spikes()

    def _compute_baseline(self):
        arr_keys = list(self._baseline_buffer[0].keys())
        mat = np.array([[f[k] for k in arr_keys] for f in self._baseline_buffer])
        self._baseline_mean = dict(zip(arr_keys, mat.mean(axis=0)))
        self._baseline_std = dict(zip(arr_keys, mat.std(axis=0) + 1e-3))
        self._baseline_ready = True

    def _detect_spikes(self) -> List[MicroEvent]:
        events: List[MicroEvent] = []
        if len(self.buf) < 3:
            return events
        # look at most recent frame vs baseline
        t_now, aus_now = self.buf[-1]
        for au, v in aus_now.items():
            mu = self._baseline_mean.get(au, 0.0)
            sd = self._baseline_std.get(au, 1.0)
            if (v - mu) / sd > self.spike_sigma and v > 0.25:
                # look back for onset — first frame where au < mu + sd
                onset_t = t_now
                for t_prev, aus_prev in reversed(self.buf):
                    if aus_prev.get(au, 0) < mu + sd:
                        onset_t = t_prev
                        break
                dur_ms = (t_now - onset_t) * 1000
                if dur_ms < self.max_duration_ms:
                    events.append(MicroEvent(au=au, peak=v,
                                             duration_ms=dur_ms, t_onset=onset_t))
        return events

    def reset_baseline(self):
        self._baseline_ready = False
        self._baseline_buffer.clear()
