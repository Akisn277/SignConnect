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
MODEL_PATH   = "./models/best_bilstm.keras"
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.npy")
FEAT_MIN     = os.path.join(MODEL_DIR, "feat_min.npy")
FEAT_RNG     = os.path.join(MODEL_DIR, "feat_rng.npy")

# 🔥 FIXED PATH
HAND_MODEL_PATH = "Sign_to_text/hand_landmarker.task"

TIMESTEPS            = 30
CONFIDENCE_THRESHOLD = 0.3
AUTO_RESET_FRAMES    = 40
# =============================================================================

# 🌍 GLOBAL OUTPUTS
latest_prediction = "Waiting for sign..."
latest_frame = None


def load_model_and_meta():
    print("[INFO] Loading model...")
    model    = tf.keras.models.load_model(MODEL_PATH, compile=False)
    classes  = np.load(CLASSES_PATH, allow_pickle=True)
    feat_min = np.load(FEAT_MIN)
    feat_rng = np.load(FEAT_RNG)
    print("[INFO] Model loaded successfully")
    return model, classes, feat_min, feat_rng


def normalize(sequence, feat_min, feat_rng):
    return ((sequence - feat_min) / feat_rng).astype(np.float32)


def run_sign_detection():
    global latest_prediction, latest_frame

    model, classes, feat_min, feat_rng = load_model_and_meta()

    # 🔥 MediaPipe setup (ONLY ONCE)
    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not opening")
        return

    frame_buffer = deque(maxlen=TIMESTEPS)
    cooldown_counter = 0
    state = "COLLECTING"

    print("[INFO] Camera started...")

    # 🔥 FIX: Move outside loop
    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame not received, retrying...")
                continue

            frame = cv2.flip(frame, 1)
            img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            result = landmarker.detect(mp_image)

            hand_present = bool(result.hand_landmarks)

            if hand_present:
                lms = result.hand_landmarks[0]

                vec = np.array(
                    [coord for lm in lms for coord in (lm.x, lm.y)],
                    dtype=np.float32
                )

                # 🔥 Draw landmarks
                h_px, w_px = frame.shape[:2]
                for lm in lms:
                    cx, cy = int(lm.x * w_px), int(lm.y * h_px)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            else:
                vec = np.zeros(42, dtype=np.float32)

            # ================= PREDICTION =================
            if state == "COLLECTING":

                if hand_present:
                    frame_buffer.append(vec)
                else:
                    frame_buffer.clear()

                if len(frame_buffer) == TIMESTEPS:
                    sequence = normalize(np.array(frame_buffer), feat_min, feat_rng)
                    X_in = sequence[np.newaxis, ...]

                    probs = model.predict(X_in, verbose=0)[0]
                    idx   = int(np.argmax(probs))
                    conf  = float(probs[idx])

                    predicted = classes[idx]
                    latest_prediction = predicted

                    print(f"[PRED] {predicted} ({conf*100:.1f}%)")

                    frame_buffer.clear()
                    cooldown_counter = AUTO_RESET_FRAMES
                    state = "COOLDOWN"

            else:
                cooldown_counter -= 1
                if cooldown_counter <= 0:
                    state = "COLLECTING"
                    frame_buffer.clear()

            # 🔥 SAVE FRAME FOR STREAMING
            latest_frame = frame.copy()

    cap.release()


def get_frame():
    global latest_frame
    return latest_frame