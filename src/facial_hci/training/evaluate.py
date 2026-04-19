"""Offline evaluation of rule-based emotion classifier on sample datasets."""
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from ..inference.emotion_rules import classify_emotion


def evaluate(samples: List[Dict]) -> Dict:
    """samples: [{'aus': {...}, 'label': 'happiness'}, ...]"""
    correct = 0
    per_label = defaultdict(lambda: [0, 0])  # [correct, total]
    confusion = Counter()
    for s in samples:
        pred, _ = classify_emotion(s["aus"])
        tot_idx = per_label[s["label"]]
        tot_idx[1] += 1
        if pred == s["label"]:
            tot_idx[0] += 1
            correct += 1
        confusion[(s["label"], pred)] += 1
    return {
        "accuracy": correct / max(1, len(samples)),
        "per_label": {k: {"correct": v[0], "total": v[1],
                          "acc": v[0]/max(1,v[1])} for k, v in per_label.items()},
        "confusion": {f"{a}->{b}": n for (a,b), n in confusion.items()},
        "n": len(samples),
    }
