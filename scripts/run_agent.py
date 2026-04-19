"""Local CLI runner — uses your local webcam directly via OpenCV.

Usage: python scripts/run_agent.py
Press 'q' to quit.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from src.facial_hci.agent import FacialHCIAgent
from src.facial_hci.features.action_units import AU_THRESHOLD


def main():
    print("Opening webcam … press 'q' to quit.")
    agent = FacialHCIAgent(session_id="cli")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            r = agent.analyze(frame)

            y = 28
            def put(txt, col=(0,255,0), size=0.7, th=2):
                nonlocal y
                cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, size, col, th)
                y += int(size * 34 + 4)

            if r.face_detected:
                put(f"Emotion: {r.emotion} ({r.emotion_confidence:.2f})")
                put(f"Cognitive: {r.cognitive_state}", (255,255,0))
                put(f"Blinks/min: {r.blink_rate_per_min:.0f}  "
                    f"Head y/p/r: {r.head_pose['yaw']:+.0f}/{r.head_pose['pitch']:+.0f}/"
                    f"{r.head_pose['roll']:+.0f}", (200,200,255), 0.55, 1)
                for au, v in sorted(r.action_units.items(), key=lambda x:-x[1])[:5]:
                    if v > AU_THRESHOLD:
                        put(f"{au}: {v:.2f}", (0,200,255), 0.5, 1)
                if r.thought_inference:
                    put(r.thought_inference[:80], (255,255,255), 0.55, 1)
            else:
                put("No face detected", (100,100,255))

            cv2.imshow("Facial HCI Agent (CLI)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        agent.close()


if __name__ == "__main__":
    main()
