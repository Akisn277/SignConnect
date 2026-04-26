from flask import Flask, jsonify, render_template, Response
import threading
import cv2

from Voice_to_text.voice_to_text import voicetotxt
from Sign_to_text import isl_inference as isl

app = Flask(__name__)

# 🔥 Start sign detection in background
def start_sign_thread():
    print("🔥 Starting sign detection thread...")
    isl.run_sign_detection()

threading.Thread(target=start_sign_thread, daemon=True).start()


# ================= API =================
@app.route('/api/live')
def live():
    return jsonify({
        "voice_text": "Listening...",
        "sign_text": isl.latest_prediction
    })


# ================= VIDEO STREAM =================
def gen_frames():
    while True:
        frame = isl.get_frame()

        if frame is None:
            print("⚠️ Waiting for frame...")
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ================= FRONTEND =================
@app.route("/")
def home():
    return render_template("index.html")


# ================= SPEECH =================
@app.route('/speech', methods=['GET'])
def speech():
    text = voicetotxt()
    return jsonify({"text": text})


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)