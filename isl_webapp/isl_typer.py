"""
ISL fingerspelling typer: recognises signs from hand landmarks and types them out as text.

Setup:
    pip install opencv-python mediapipe==0.10.14 scikit-learn "numpy<2"
    pip install pyautogui        (optional: lets it type into other apps, e.g. Notepad)

Steps:
    python isl_typer.py collect a        # SPACE = start/stop recording, Q = quit.  Repeat for every letter.
    python isl_typer.py collect space    # a gesture YOU choose for "space"
    python isl_typer.py collect del      # a gesture YOU choose for "backspace" (hold it to keep deleting)
    python isl_typer.py collect rest     # relaxed hand: stops accidental typing between signs
    python isl_typer.py train            # takes a few seconds
    python isl_typer.py run              # hold a sign steady to type it

Keys while running:
    T = toggle typing into the active window     C = clear text      Q = quit
    - / = = make the hold time shorter / longer  Backspace key = delete last letter
"""
import csv
import os
import pickle
import sys
import time
from collections import Counter

import cv2
import numpy as np
import mediapipe as mp

DATA_FILE = "isl_landmarks.csv"
MODEL_FILE = "isl_model.pkl"
HOLD_SECONDS = 0.4          # how long a sign must be held steady before it is typed (change live with - and =)
MIN_CONF = 0.6              # minimum model confidence to accept a prediction
REARM_SECONDS = 0.3         # hand must be away this long before the same letter can be typed again
DEL_REPEAT_SECONDS = 0.7    # holding "del" deletes again every this many seconds
PROMPTS = ["centre", "closer", "farther", "move left", "move right",
           "move up", "move down", "tiny tilt"]   # shown while recording, to get varied samples
PROMPT_SECONDS = 2.5

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def make_hands():
    return mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                          min_tracking_confidence=0.6)


def extract(result):
    """Hand landmarks -> 129 numbers: shape of two hands + position of one hand relative to the other.
    Independent of where you stand, how far you are, and the background. None if no hand is visible."""
    if not result.multi_hand_landmarks:
        return None
    found = []
    for hl in result.multi_hand_landmarks[:2]:
        pts = np.array([[l.x, l.y, l.z] for l in hl.landmark])
        wrist = pts[0].copy()
        rel = pts - wrist
        scale = float(np.abs(rel).max()) or 1e-6
        found.append((wrist, rel / scale, scale))
    found.sort(key=lambda h: h[0][0])          # leftmost hand on screen first
    feats = np.zeros(129)
    feats[:63] = found[0][1].flatten()
    if len(found) == 2:
        feats[63:126] = found[1][1].flatten()
        feats[126:129] = (found[1][0] - found[0][0]) / max(found[0][2], found[1][2])
    return feats


def read(cap, hands):
    ok, frame = cap.read()
    if not ok:
        return None, None, None
    frame = cv2.flip(frame, 1)                 # mirror view (same in collect and run)
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        for hl in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
    return frame, res, extract(res)


def collect(label):
    label = label.lower()
    cap = cv2.VideoCapture(0)
    hands = make_hands()
    recording, count, i, rec_start = False, 0, 0, 0.0
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        while True:
            frame, res, feats = read(cap, hands)
            if frame is None:
                break
            i += 1
            if recording and feats is not None and i % 2 == 0:
                writer.writerow([label, *np.round(feats, 5)])
                count += 1
            colour = (0, 0, 255) if recording else (0, 255, 0)
            cv2.putText(frame, f"'{label}'  {'RECORDING' if recording else 'paused'}  saved: {count}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
            cv2.putText(frame, "SPACE = record/stop, Q = quit", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if recording:   # keep the sign, but follow the prompt so the samples cover many poses
                step = int((time.time() - rec_start) / PROMPT_SECONDS) % len(PROMPTS)
                cv2.putText(frame, PROMPTS[step].upper(), (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.imshow("Collect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:
                recording = not recording
                if recording:
                    rec_start = time.time()
    cap.release()
    cv2.destroyAllWindows()


def train():
    from sklearn.ensemble import RandomForestClassifier

    if not os.path.exists(DATA_FILE):
        sys.exit("No data yet. Run the collect step first.")
    rows = list(csv.reader(open(DATA_FILE)))
    y = np.array([r[0] for r in rows])
    X = np.array([[float(v) for v in r[1:]] for r in rows])

    counts = Counter(y.tolist())
    print("Samples per sign:", dict(sorted(counts.items())))
    for needed in ("space", "del", "rest"):
        if needed not in counts:
            print(f"NOTE: no '{needed}' samples recorded yet (python isl_typer.py collect {needed})")
    few = [k for k, v in counts.items() if v < 50]
    if few:
        print("WARNING: fewer than 50 samples for:", few)

    # Fair test: cut each sign's recording into ~3-second blocks and keep every 5th block for testing.
    # (A random split tests on frames almost identical to training frames; testing only on the end of
    #  each recording is too harsh. Whole blocks held out sits in between.)
    BLOCK = 40
    train_idx, test_idx = [], []
    for lab in sorted(counts):
        idx = np.where(y == lab)[0]
        if len(idx) < 3 * BLOCK:                       # too short for blocks: use the last 20%
            cut = int(len(idx) * 0.8)
            train_idx += idx[:cut].tolist()
            test_idx += idx[cut:].tolist()
            continue
        is_test = (np.arange(len(idx)) // BLOCK) % 5 == 4
        if not is_test.any():
            is_test = (np.arange(len(idx)) // BLOCK) == (len(idx) - 1) // BLOCK
        test_idx += idx[is_test].tolist()
        train_idx += idx[~is_test].tolist()
    X_tr, y_tr, X_te, y_te = X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=0)   # 100 trees = fast live predictions
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    print(f"Test accuracy: {(pred == y_te).mean():.1%}  ({len(y_te)} frames from held-out blocks of each recording)")
    errors = Counter((a, b) for a, b in zip(y_te.tolist(), pred.tolist()) if a != b)
    if errors:
        print("Most confused (true -> predicted):")
        for (a, b), n in errors.most_common(8):
            print(f"   {a} -> {b}: {n} frames")

    model.fit(X, y)                            # final model uses all the data
    pickle.dump(model, open(MODEL_FILE, "wb"))
    print("Saved", MODEL_FILE)


class TypingLogic:
    """Decides when a held sign should be typed. Uses real time, so it behaves the same at any frame rate."""

    def __init__(self, hold_s=HOLD_SECONDS):
        self.hold_s = hold_s
        self.streak_label = None
        self.streak_start = None
        self.idle_start = None
        self.armed = True
        self.last_typed = None
        self.last_commit = 0.0

    def held(self, now):
        return (now - self.streak_start) if self.streak_start is not None else 0.0

    def update(self, label, now):
        """label = sign currently seen (or None). Returns the label to type right now, or None."""
        if label is None or label != self.streak_label:
            self.streak_label = label
            self.streak_start = now if label else None
        if label is None:
            if self.idle_start is None:
                self.idle_start = now
            if now - self.idle_start >= REARM_SECONDS:
                self.armed = True
            return None
        self.idle_start = None
        if now - self.streak_start < self.hold_s:
            return None
        repeat = label == "del" and now - self.last_commit >= DEL_REPEAT_SECONDS
        if self.armed or label != self.last_typed or repeat:
            self.armed, self.last_typed, self.last_commit = False, label, now
            return label
        return None


def run():
    try:
        import pyautogui
        pyautogui.PAUSE = 0                    # no built-in delay after each key press
    except ImportError:
        pyautogui = None

    model = pickle.load(open(MODEL_FILE, "rb"))
    cap = cv2.VideoCapture(0)
    hands = make_hands()

    logic = TypingLogic()
    text = ""
    typing = False
    message = ""
    fps, prev = 0.0, time.time()

    while True:
        frame, res, feats = read(cap, hands)
        if frame is None:
            break
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-3)
        prev = now

        label, top, conf = None, None, 0.0
        if feats is not None:
            probs = model.predict_proba(feats.reshape(1, -1))[0]
            j = int(probs.argmax())
            top, conf = model.classes_[j], float(probs[j])
            if conf >= MIN_CONF and top != "rest":
                label = top

        typed = logic.update(label, now)
        if typed:
            if typed == "space":
                text += " "
                key = "space"
            elif typed == "del":
                text = text[:-1]
                key = "backspace"
            else:
                text += typed.upper()
                key = typed.lower()
            if typing and pyautogui:
                pyautogui.press(key)

        # on-screen display
        h, w = frame.shape[:2]
        if label:
            cv2.putText(frame, label.upper(), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 0), 5)
            bar = int(min(logic.held(now) / logic.hold_s, 1.0) * 200)
            cv2.rectangle(frame, (10, 85), (10 + bar, 100), (0, 255, 0), -1)
            cv2.rectangle(frame, (10, 85), (210, 100), (255, 255, 255), 1)
        sees = f"sees: {top} {conf:.0%}" if top else "no hand"
        cv2.putText(frame, sees, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, text[-40:] + "_", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.putText(frame, f"Typing into other apps: {'ON' if typing else 'off'} (T)   Clear: C   Quit: Q",
                    (10, h - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Hold time: {logic.hold_s:.1f}s (- / =)   {fps:.0f} fps   {message}",
                    (10, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("ISL typer", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("c"):
            text = ""
        elif k == 8:                           # Backspace key on the keyboard
            text = text[:-1]
        elif k == ord("-"):
            logic.hold_s = max(0.1, logic.hold_s - 0.1)
        elif k in (ord("="), ord("+")):
            logic.hold_s = min(2.0, logic.hold_s + 0.1)
        elif k == ord("t"):
            if pyautogui is None:
                message = "pip install pyautogui first"
            else:
                typing = not typing
                message = ""

    cap.release()
    cv2.destroyAllWindows()
    if text:
        print("Final text:", text)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "collect" and len(sys.argv) > 2:
        collect(sys.argv[2])
    elif cmd == "train":
        train()
    elif cmd == "run":
        run()
    else:
        print(__doc__)
