"""
ISL fingerspelling web app.

The browser opens your webcam and sends small frames to this server. The server finds the hand
with MediaPipe, classifies it with your trained model, and sends the letter back. The browser
turns held signs into text.

Run locally:  python app.py   then open http://localhost:5000
On Render:    gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
Needs (same folder): best.pt (YOLO model)  and/or  isl_model.pkl + isl_typer.py (landmark model)
"""
import base64
import os
import pickle
import sys
import threading

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, jsonify, render_template, request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

app = Flask(__name__)
lock = threading.Lock()   # MediaPipe's Hands object is not thread-safe
hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                                 min_tracking_confidence=0.6)

models = {}   # name -> dict(predict=callable)


def load_models():
    yolo_path = os.path.join(HERE, "best.pt")
    if os.path.exists(yolo_path):
        try:
            from ultralytics import YOLO
            models["yolo"] = {"label": "YOLO (image crop)", "obj": YOLO(yolo_path)}
            print("Loaded YOLO model")
        except ImportError:
            print("ultralytics not installed, skipping best.pt")

    pkl_path = os.path.join(HERE, "isl_model.pkl")
    if os.path.exists(pkl_path) and os.path.exists(os.path.join(HERE, "isl_typer.py")):
        from isl_typer import extract
        with open(pkl_path, "rb") as f:
            clf = pickle.load(f)
        models["landmarks"] = {"label": "Landmarks (random forest)", "obj": clf, "extract": extract}
        print("Loaded landmark model")

    if not models:
        sys.exit("No model found. Put best.pt and/or isl_model.pkl + isl_typer.py next to app.py.")


def square_box(xs, ys, w, h):
    """Square, padded box around the hand(s), same as run_live_isl.py. Pixel coordinates."""
    cx, cy = (min(xs) + max(xs)) / 2 * w, (min(ys) + max(ys)) / 2 * h
    side = max((max(xs) - min(xs)) * w, (max(ys) - min(ys)) * h) * 1.3
    x1, x2 = int(max(cx - side / 2, 0)), int(min(cx + side / 2, w))
    y1, y2 = int(max(cy - side / 2, 0)), int(min(cy + side / 2, h))
    return x1, y1, x2, y2


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/info")
def info():
    # landmark model first: it is the one isl_typer.py uses, so it becomes the default choice
    order = sorted(models, key=lambda k: k != "landmarks")
    return jsonify(models=[{"id": k, "label": models[k]["label"]} for k in order])


@app.post("/predict")
def predict():
    data = request.get_json(force=True, silent=True) or {}
    name = data.get("model")
    if name not in models:
        return jsonify(error="unknown model"), 400
    try:
        raw = base64.b64decode(data["image"].split(",")[-1])
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        frame = None
    if frame is None:
        return jsonify(error="bad image"), 400

    h, w = frame.shape[:2]
    # The landmark model was trained on a mirrored view; the YOLO pipeline uses the raw view.
    mirrored = name == "landmarks"
    img = cv2.flip(frame, 1) if mirrored else frame

    with lock:
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return jsonify(hand=False)

    xs = [l.x for hl in res.multi_hand_landmarks for l in hl.landmark]
    ys = [l.y for hl in res.multi_hand_landmarks for l in hl.landmark]
    x1, y1, x2, y2 = square_box(xs, ys, w, h)

    label, conf = None, 0.0
    m = models[name]
    if name == "yolo":
        crop = img[y1:y2, x1:x2]
        if crop.size:
            r = m["obj"](crop, verbose=False)[0]
            label, conf = r.names[int(r.probs.top1)], float(r.probs.top1conf)
    else:
        feats = m["extract"](res)
        if feats is not None:
            probs = m["obj"].predict_proba(feats.reshape(1, -1))[0]
            j = int(probs.argmax())
            label, conf = str(m["obj"].classes_[j]), float(probs[j])

    if mirrored:   # report the box in the raw (un-mirrored) frame so the page can draw it
        x1, x2 = w - x2, w - x1
    return jsonify(hand=True, label=(label or "").lower() or None, conf=conf,
                   box=[x1 / w, y1 / h, x2 / w, y2 / h])


load_models()   # runs on import, so gunicorn (app:app) works as well as "python app.py"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0" if "PORT" in os.environ else "127.0.0.1", port=port, threaded=True)
