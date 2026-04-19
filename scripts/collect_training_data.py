"""Interactive CLI to collect labeled samples and train a personal classifier.

Usage: python scripts/collect_training_data.py <session_id>
Then press keys: n=neutral, h=happy, s=sad, f=focused, c=confused, t=train, q=quit
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
from src.facial_hci.agent import FacialHCIAgent
from src.facial_hci.training.personalize import PersonalProfile

KEYMAP = {
    ord('n'): "neutral", ord('h'): "happy", ord('s'): "sad",
    ord('f'): "focused", ord('c'): "confused", ord('a'): "angry",
}


def main():
    session_id = sys.argv[1] if len(sys.argv) > 1 else "default_user"
    profile = PersonalProfile.load_or_new(session_id)
    agent = FacialHCIAgent(session_id=session_id, personalization=profile)

    cap = cv2.VideoCapture(0)
    print("Keys: n/h/s/f/c/a = tag label, t = train, q = quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            r = agent.analyze(frame)
            msg = f"samples={profile.sample_count()}  emotion={r.emotion}"
            cv2.putText(frame, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Collect training data", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord('t'):
                res = profile.train_classifier()
                print("TRAIN:", res)
                profile.save()
            elif k in KEYMAP:
                profile.collect_sample(KEYMAP[k], agent)
                profile.save()
                print(f"+ {KEYMAP[k]}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        agent.close()
        print(f"Saved to user_profiles/{session_id}/")


if __name__ == "__main__":
    main()
