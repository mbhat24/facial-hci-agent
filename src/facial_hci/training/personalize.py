"""Per-user personalization.

Stores a running baseline of AU means/std for each user, plus optionally
fine-tunes a lightweight classifier (LogisticRegression) on user-tagged
samples for a personalized emotion/state head.

Storage: user_profiles/<session_id>/profile.json + classifier.joblib
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

from ..config import settings
from ..logging_utils import get_logger

log = get_logger(__name__)

AU_KEYS = [
    "AU1","AU2","AU4","AU5","AU6","AU7","AU9","AU10","AU12","AU14",
    "AU15","AU17","AU20","AU23","AU24","AU25","AU26","AU28","AU43","AU45",
]


@dataclass
class PersonalProfile:
    session_id: str
    baseline_mean: Dict[str, float] = field(default_factory=dict)
    baseline_std: Dict[str, float] = field(default_factory=dict)
    samples: List[Dict] = field(default_factory=list)   # {label, aus, ts}
    frame_count: int = 0
    created_at: float = field(default_factory=time.time)

    # runtime — not serialized
    _classifier: Optional[object] = field(default=None, repr=False)
    _scaler: Optional[object] = field(default=None, repr=False)

    # ---------- baseline running stats ----------
    def update_baseline(self, aus: Dict[str, float]):
        self.frame_count += 1
        n = self.frame_count
        for k, v in aus.items():
            prev_mean = self.baseline_mean.get(k, 0.0)
            prev_var = self.baseline_std.get(k, 0.0) ** 2
            new_mean = prev_mean + (v - prev_mean) / n
            new_var = prev_var + ((v - prev_mean) * (v - new_mean) - prev_var) / n
            self.baseline_mean[k] = new_mean
            self.baseline_std[k] = float(np.sqrt(max(new_var, 0)))

    def adjust_aus(self, aus: Dict[str, float]) -> Dict[str, float]:
        """Subtract per-user baseline — highlights *deviations* from neutral."""
        self.update_baseline(aus)
        if self.frame_count < 30:
            return aus
        adj = {}
        for k, v in aus.items():
            m = self.baseline_mean.get(k, 0.0)
            adj[k] = float(max(0.0, v - m * 0.7))    # partial subtraction
        return adj

    # ---------- sample collection & personalized classifier ----------
    def collect_sample(self, label: str, agent) -> None:
        """Collect a labeled sample from the agent's current state."""
        # use the most recent AU vector (from the micro-detector buffer if available)
        if not agent.micro_detector.buf:
            return
        _, aus = agent.micro_detector.buf[-1]
        self.samples.append({
            "label": label,
            "aus": {k: float(aus.get(k, 0.0)) for k in AU_KEYS},
            "ts": time.time(),
        })
        log.info(f"Collected sample '{label}' ({len(self.samples)} total)")

    def sample_count(self) -> int:
        return len(self.samples)

    def train_classifier(self) -> dict:
        """Train a per-user LogisticRegression on collected samples.

        Returns a dict with metrics or error info.
        """
        if not SKLEARN_OK:
            return {"ok": False, "error": "scikit-learn not installed"}
        if len(self.samples) < 8:
            return {"ok": False, "error": "need >= 8 samples"}

        X = np.array([[s["aus"][k] for k in AU_KEYS] for s in self.samples])
        y = np.array([s["label"] for s in self.samples])
        n_classes = len(set(y))
        if n_classes < 2:
            return {"ok": False, "error": "need >= 2 distinct labels"}

        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._classifier = LogisticRegression(max_iter=500, multi_class="auto").fit(Xs, y)
        train_acc = float(self._classifier.score(Xs, y))
        log.info(f"Personal classifier trained ({n_classes} classes, acc={train_acc:.2f})")
        return {"ok": True, "train_acc": train_acc,
                "classes": sorted(set(y)), "n_samples": len(y)}

    def predict(self, aus: Dict[str, float]) -> Optional[Dict[str, float]]:
        if self._classifier is None or self._scaler is None:
            return None
        x = np.array([[aus.get(k, 0.0) for k in AU_KEYS]])
        xs = self._scaler.transform(x)
        probs = self._classifier.predict_proba(xs)[0]
        labels = self._classifier.classes_
        return {lbl: float(p) for lbl, p in zip(labels, probs)}

    # ---------- persistence ----------
    def _dir(self) -> Path:
        d = settings.profiles_dir / self.session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self):
        d = self._dir()
        payload = {
            "session_id": self.session_id,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "samples": self.samples,
            "frame_count": self.frame_count,
            "created_at": self.created_at,
        }
        (d / "profile.json").write_text(json.dumps(payload, indent=2))
        if self._classifier is not None and SKLEARN_OK:
            joblib.dump({"clf": self._classifier, "scaler": self._scaler},
                        d / "classifier.joblib")

    @classmethod
    def load_or_new(cls, session_id: str) -> "PersonalProfile":
        d = settings.profiles_dir / session_id
        p = d / "profile.json"
        if not p.exists():
            return cls(session_id=session_id)
        try:
            data = json.loads(p.read_text())
            prof = cls(
                session_id=data["session_id"],
                baseline_mean=data.get("baseline_mean", {}),
                baseline_std=data.get("baseline_std", {}),
                samples=data.get("samples", []),
                frame_count=data.get("frame_count", 0),
                created_at=data.get("created_at", time.time()),
            )
            clf_path = d / "classifier.joblib"
            if clf_path.exists() and SKLEARN_OK:
                blob = joblib.load(clf_path)
                prof._classifier = blob["clf"]
                prof._scaler = blob["scaler"]
            return prof
        except Exception as e:
            log.warning(f"Profile load failed ({e}) — starting fresh.")
            return cls(session_id=session_id)
