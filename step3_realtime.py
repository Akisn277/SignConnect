"""
=============================================================================
STEP 3 — Real-Time Sign Language to Text
=============================================================================
Uses your trained BiLSTM model + webcam to predict ISL gestures in real time.

Controls:
    Q  →  Quit
    C  →  Clear the sentence on screen

Run:
    python step3_realtime.py
=============================================================================
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# =============================================================================
# CONFIG
# =============================================================================
MODEL_DIR    = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_bilstm.h5")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.npy")
FEAT_MIN     = os.path.join(MODEL_DIR, "feat_min.npy")
FEAT_RNG     = os.path.join(MODEL_DIR, "feat_rng.npy")

TIMESTEPS        = 30       # must match training
CONFIDENCE_THRESHOLD = 0.5 # minimum confidence to show prediction
COOLDOWN_FRAMES  = 5      # frames to wait before next prediction
# =============================================================================


def load_model_and_meta():
    print("[INFO] Loading model …")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    feat_min = np.load(FEAT_MIN)
    feat_rng = np.load(FEAT_RNG)
    print(f"[INFO] Model loaded — {len(classes)} classes")
    return model, classes, feat_min, feat_rng


def normalize(sequence, feat_min, feat_rng):
    N, F = sequence.shape
    return ((sequence - feat_min) / feat_rng).astype(np.float32)


def draw_ui(frame, predicted_text, confidence, sentence, buffer_pct):
    h, w = frame.shape[:2]

    # ── Semi-transparent top bar ──────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # ── Title ─────────────────────────────────────────────────────────────────
    cv2.putText(frame, "ISL Sign → Text", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ── Controls hint ─────────────────────────────────────────────────────────
    cv2.putText(frame, "Q: Quit  |  C: Clear", (w - 220, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    # ── Buffer progress bar ───────────────────────────────────────────────────
    bar_w = int(w * buffer_pct)
    cv2.rectangle(frame, (0, 70), (w, 80), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, 70), (bar_w, 80), (0, 200, 100), -1)

    # ── Predicted gesture ─────────────────────────────────────────────────────
    if predicted_text:
        conf_color = (0, 255, 100) if confidence >= 0.75 else (0, 200, 255)
        label = f"{predicted_text}  ({confidence*100:.0f}%)"
        cv2.putText(frame, label, (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, conf_color, 3)

    # ── Sentence box ─────────────────────────────────────────────────────────
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 55), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)
    sentence_display = " ".join(sentence) if sentence else "—"
    # Truncate if too long
    if len(sentence_display) > 60:
        sentence_display = "… " + sentence_display[-58:]
    cv2.putText(frame, sentence_display, (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return frame


def main():
    model, classes, feat_min, feat_rng = load_model_and_meta()

    # ── MediaPipe setup ───────────────────────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, "hand_landmarker.task")
    if not os.path.exists(model_path):
        print("[INFO] Downloading hand landmarker model …")
        import urllib.request
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        urllib.request.urlretrieve(url, model_path)

    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── State ─────────────────────────────────────────────────────────────────
    frame_buffer  = deque(maxlen=TIMESTEPS)  # rolling window of landmark vecs
    sentence      = []                        # accumulated words
    predicted     = ""
    confidence    = 0.0
    cooldown      = 0
    last_word     = ""

    print("\n[INFO] Webcam started — show your hand gestures!")
    print("[INFO] Press Q to quit, C to clear sentence\n")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)   # mirror view
            img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result   = landmarker.detect(mp_image)

            # ── Extract landmarks ─────────────────────────────────────────────
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                vec = np.array(
                    [coord for lm in lms for coord in (lm.x, lm.y)],
                    dtype=np.float32
                )
                # Draw hand landmarks on frame
                h, w = frame.shape[:2]
                for lm in lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 200), -1)
            else:
                vec = np.zeros(42, dtype=np.float32)

            frame_buffer.append(vec)

            # ── Predict when buffer is full ───────────────────────────────────
            buffer_pct = len(frame_buffer) / TIMESTEPS

            if len(frame_buffer) == TIMESTEPS and cooldown == 0 and result.hand_landmarks:
                sequence  = np.array(frame_buffer)           # (30, 42)
                sequence  = normalize(sequence, feat_min, feat_rng)
                X_in      = sequence[np.newaxis, ...]         # (1, 30, 42)

                probs     = model.predict(X_in, verbose=0)[0]
                top_idx   = int(np.argmax(probs))
                top_conf  = float(probs[top_idx])

                if top_conf >= CONFIDENCE_THRESHOLD:
                    predicted  = classes[top_idx]
                    confidence = top_conf

                    # Add to sentence if different from last word
                    if predicted != last_word:
                        sentence.append(predicted)
                        last_word = predicted
                        if len(sentence) > 10:
                            sentence.pop(0)

                    cooldown = COOLDOWN_FRAMES
                else:
                    predicted  = ""
                    confidence = 0.0

            if cooldown > 0:
                cooldown -= 1

            # ── Draw UI ───────────────────────────────────────────────────────
            frame = draw_ui(frame, predicted, confidence, sentence, buffer_pct)
            cv2.imshow("ISL Sign Language → Text", frame)

            # ── Key controls ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence.clear()
                last_word  = ""
                predicted  = ""
                confidence = 0.0

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
