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
MODEL_PATH   = os.path.join(MODEL_DIR, "best_bilstm.h5")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.npy")
FEAT_MIN     = os.path.join(MODEL_DIR, "feat_min.npy")
FEAT_RNG     = os.path.join(MODEL_DIR, "feat_rng.npy")

TIMESTEPS            = 30
CONFIDENCE_THRESHOLD = 0.55

# Frames to wait after a prediction before accepting next sign
# At ~30fps: 30 = 1 second, 45 = 1.5 seconds, 60 = 2 seconds
AUTO_RESET_FRAMES = 40
# =============================================================================


def load_model_and_meta():
    print("[INFO] Loading model ...")
    model    = tf.keras.models.load_model(MODEL_PATH, compile=False)
    classes  = np.load(CLASSES_PATH, allow_pickle=True)
    feat_min = np.load(FEAT_MIN)
    feat_rng = np.load(FEAT_RNG)
    print(f"[INFO] Model loaded -- {len(classes)} classes")
    return model, classes, feat_min, feat_rng


def normalize(sequence, feat_min, feat_rng):
    return ((sequence - feat_min) / feat_rng).astype(np.float32)


def draw_ui(frame, predicted_text, confidence, sentence, buffer_pct, state_label, state_color):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "ISL Sign -> Text", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Q:Quit C:Clear | {state_label}", (w - 320, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, state_color, 1)

    # Buffer / cooldown progress bar
    bar_w = int(w * buffer_pct)
    cv2.rectangle(frame, (0, 70), (w, 80), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, 70), (bar_w, 80), state_color, -1)

    if predicted_text:
        conf_color = (0, 255, 100) if confidence >= 0.75 else (0, 200, 255)
        cv2.putText(frame, f"{predicted_text}  ({confidence*100:.0f}%)",
                    (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, conf_color, 3)

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 55), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)
    sentence_display = " ".join(sentence) if sentence else "--"
    if len(sentence_display) > 60:
        sentence_display = "... " + sentence_display[-58:]
    cv2.putText(frame, sentence_display, (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return frame


def main():
    model, classes, feat_min, feat_rng = load_model_and_meta()

    model_path = os.path.join(MODEL_DIR, "hand_landmarker.task")
    if not os.path.exists(model_path):
        print("[INFO] Downloading hand landmarker model ...")
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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # States
    STATE_COLLECTING = "COLLECTING"
    STATE_COOLDOWN   = "COOLDOWN"

    frame_buffer    = deque(maxlen=TIMESTEPS)
    sentence        = []
    predicted       = ""
    confidence      = 0.0
    state           = STATE_COLLECTING
    cooldown_counter = 0

    print("\n[INFO] Ready! Show a sign and hold it for 1 second.")
    print("[INFO] After prediction, wait ~1.5s for next sign automatically.\n")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result   = landmarker.detect(mp_image)

            hand_present = bool(result.hand_landmarks)

            if hand_present:
                lms = result.hand_landmarks[0]
                vec = np.array(
                    [coord for lm in lms for coord in (lm.x, lm.y)],
                    dtype=np.float32
                )
                h_px, w_px = frame.shape[:2]
                for lm in lms:
                    cx, cy = int(lm.x * w_px), int(lm.y * h_px)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 200), -1)
            else:
                vec = np.zeros(42, dtype=np.float32)

            # ── COLLECTING: fill buffer and predict ───────────────────────────
            if state == STATE_COLLECTING:

                if hand_present:
                    frame_buffer.append(vec)
                else:
                    # Hand left mid-sign — reset so partial frames don't pollute
                    frame_buffer.clear()

                buffer_pct = len(frame_buffer) / TIMESTEPS

                if len(frame_buffer) == TIMESTEPS:
                    sequence = normalize(np.array(frame_buffer), feat_min, feat_rng)
                    X_in     = sequence[np.newaxis, ...]
                    probs    = model.predict(X_in, verbose=0)[0]
                    top_idx  = int(np.argmax(probs))
                    top_conf = float(probs[top_idx])

                    if top_conf >= CONFIDENCE_THRESHOLD:
                        predicted  = classes[top_idx]
                        confidence = top_conf
                        if not sentence or sentence[-1] != predicted:
                            sentence.append(predicted)
                            if len(sentence) > 10:
                                sentence.pop(0)
                        print(f"[PRED] {predicted}  ({top_conf*100:.1f}%)")
                    else:
                        predicted  = ""
                        confidence = 0.0
                        print(f"[SKIP] Low confidence ({top_conf*100:.1f}%)")

                    # Wipe buffer + enter cooldown — auto resets after N frames
                    frame_buffer.clear()
                    cooldown_counter = AUTO_RESET_FRAMES
                    state = STATE_COOLDOWN

                state_label = "COLLECTING"
                state_color = (0, 220, 80)

            # ── COOLDOWN: fixed timer, no user action needed ──────────────────
            else:
                cooldown_counter -= 1
                pct_done    = 1.0 - (cooldown_counter / AUTO_RESET_FRAMES)
                buffer_pct  = pct_done   # bar drains as cooldown runs
                state_label = f"NEXT SIGN IN... {cooldown_counter}"
                state_color = (0, 140, 255)

                if cooldown_counter <= 0:
                    state = STATE_COLLECTING
                    frame_buffer.clear()
                    print("[INFO] Ready for next sign")

            # ── Draw & show ───────────────────────────────────────────────────
            frame = draw_ui(frame, predicted, confidence,
                            sentence, buffer_pct, state_label, state_color)
            cv2.imshow("ISL Sign Language -> Text", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence.clear()
                predicted        = ""
                confidence       = 0.0
                state            = STATE_COLLECTING
                cooldown_counter = 0
                frame_buffer.clear()
            # SPACE = skip cooldown manually — useful for fast signing
            elif key == ord(' '):
                if state == STATE_COOLDOWN:
                    state            = STATE_COLLECTING
                    cooldown_counter = 0
                    frame_buffer.clear()
                    print("[INFO] Cooldown skipped — ready!")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
