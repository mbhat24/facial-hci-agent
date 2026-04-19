"""Blink detection + rate tracking."""
from collections import deque
import time


class BlinkTracker:
    """Detect blinks via AU45 (eyeBlink) + track rate over 60 s window."""

    def __init__(self, threshold: float = 0.5, refractory: float = 0.15):
        self.threshold = threshold
        self.refractory = refractory   # min gap between blinks (s)
        self._blinks = deque(maxlen=500)
        self._last_state = False

    def update(self, au45: float) -> bool:
        now = time.time()
        is_closed = au45 > self.threshold
        blink_now = False
        if is_closed and not self._last_state:
            if not self._blinks or (now - self._blinks[-1]) > self.refractory:
                self._blinks.append(now)
                blink_now = True
        self._last_state = is_closed
        return blink_now

    def rate_per_minute(self) -> float:
        now = time.time()
        recent = [t for t in self._blinks if now - t < 60]
        return float(len(recent))

    def reset(self):
        self._blinks.clear()
        self._last_state = False
