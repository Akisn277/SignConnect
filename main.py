from flask import Flask, jsonify, render_template, Response
import threading
import cv2

from Voice_to_text.voice_to_text import voicetotxt
import Sign_to_text.isl_inference as isl

app = Flask(__name__)

# 🔥 Start sign detection in background (FIXED)
threading.Thread(target=isl.run_sign_detection, daemon=True).start()


# ================= API =================
@app.route('/api/live')
def live():
    return jsonify({
        "voice_text": "Listening...",
        "sign_text": isl.latest_prediction  # ✅ FIXED
    })


# ================= VIDEO STREAM =================
def gen_frames():
    while True:
        frame = isl.get_frame()  # ✅ FIXED

        if frame is None:
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