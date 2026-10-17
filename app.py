import cv2
import numpy as np
import threading
import time
from flask import Flask, Response, render_template, jsonify, request
from filters import apply_filter

app = Flask(__name__)

# ─── Global state ────────────────────────────────────────────
_lock        = threading.Lock()
_active_filter = "none"
_latest_frame  = None   # raw BGR frame for snapshot
_filtered_frame = None  # filtered BGR frame

# ─── Face detector ───────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ─── Camera thread ───────────────────────────────────────────
def camera_thread():
    global _latest_frame, _filtered_frame, _active_filter
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # mirror for natural feel
        frame = cv2.flip(frame, 1)

        # detect faces
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        # apply active filter
        with _lock:
            fname = _active_filter
        filtered = apply_filter(frame, fname, faces)

        with _lock:
            _latest_frame   = frame.copy()
            _filtered_frame = filtered.copy()

    cap.release()


# ─── MJPEG stream generator ──────────────────────────────────
def mjpeg_generator():
    while True:
        with _lock:
            frame = _filtered_frame
        if frame is None:
            time.sleep(0.03)
            continue
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buf.tobytes() +
            b"\r\n"
        )
        time.sleep(1 / 30)


# ─── Routes ──────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/set_filter", methods=["POST"])
def set_filter():
    data = request.get_json(force=True)
    name = data.get("filter", "none")
    with _lock:
        global _active_filter
        _active_filter = name
    return jsonify({"status": "ok", "filter": name})


@app.route("/snapshot")
def snapshot():
    with _lock:
        frame = _filtered_frame
    if frame is None:
        return "No frame", 503
    ret, buf = cv2.imencode(".png", frame)
    if not ret:
        return "Encode error", 500
    return Response(
        buf.tobytes(),
        mimetype="image/png",
        headers={"Content-Disposition": 'attachment; filename="facefilter_snapshot.png"'}
    )


# ─── Start ───────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()
    print("\n  *** FaceFilter AI  -  http://localhost:5000 ***\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
