import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR     = r"C:\Users\vidhy\OneDrive - SRM Institute of Science & Technology\Desktop\signconnect"
MODEL_DIR    = os.path.join(BASE_DIR, "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "best_bilstm.keras")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.npy")
FEAT_MIN     = os.path.join(MODEL_DIR, "feat_min.npy")
FEAT_RNG     = os.path.join(MODEL_DIR, "feat_rng.npy")

TIMESTEPS            = 20
CONFIDENCE_THRESHOLD = 0.3
AUTO_RESET_FRAMES    = 40

# =============================================================================
# SIGN → SENTENCE MAP (matches your exact classes.npy labels)
# =============================================================================
SIGN_SENTENCES = {
    # ── Phrases ───────────────────────────────────────────────────────────────
    "again"           : "Could you please say that again?",
    "agree"           : "Yes, I completely agree with you!",
    "answer"          : "I know the answer!",
    "attendance"      : "Please mark my attendance.",
    "book"            : "Can I borrow your book?",
    "break"           : "Can we take a short break?",
    "change"          : "I would like to make a change.",
    "chat"            : "Let us have a quick chat.",
    "congratulations" : "Congratulations, well done!",
    "email"           : "Please send me an email.",
    "file"            : "Can you share that file with me?",
    "good morning"    : "Good morning! Hope you have a wonderful day!",
    "happy birthday"  : "Happy Birthday! Wishing you all the best!",
    "home"            : "I am going home now.",
    "how are you"     : "How are you doing today?",
    "i need help"     : "I need some help please!",
    "join"            : "Can I join you?",
    "keepsmile"       : "Keep smiling, everything will be okay!",
    "meet"            : "Nice to meet you!",
    "mistake"         : "I think I made a mistake, sorry!",
    "open"            : "Could you please open that?",
    "opinion"         : "What is your opinion on this?",
    "pass"            : "Could you pass that to me please?",
    "please"          : "Could you please help me?",
    "practice"        : "We need to practice more.",
    "pressure"        : "I am feeling a lot of pressure.",
    "problem"         : "We have a problem, can you help?",
    "questions"       : "I have a few questions please.",
    "remember"        : "Please remember to do that.",
    "seat"            : "Please take your seat.",
    "shift"           : "It is time for the shift change.",
    "sick"            : "I am not feeling well today.",
    "stop"            : "Please stop right there.",
    "sun"             : "The sun is shining beautifully today!",
    "team"            : "Great job everyone, we are a great team!",
    "thirsty"         : "I am thirsty, can I have some water?",
    "this"            : "Please take a look at this.",
    "together"        : "Let us do this together!",
    "understand"      : "I understand, thank you for explaining.",
    "wait"            : "Please wait for a moment.",
    "where"           : "Where are you right now?",

    # ── Alphabets ─────────────────────────────────────────────────────────────
    "alpha_a"  : "The letter A.",
    "alpha_b"  : "The letter B.",
    "alpha_c"  : "The letter C.",
    "alpha_d"  : "The letter D.",
    "alpha_e"  : "The letter E.",
    "alpha_f"  : "The letter F.",
    "alpha_g"  : "The letter G.",
    "alpha_i"  : "The letter I.",
    "alpha_k"  : "The letter K.",
    "alpha_l"  : "The letter L.",
    "alpha_m"  : "The letter M.",
    "alpha_n"  : "The letter N.",
    "alpha_o"  : "The letter O.",
    "alpha_p"  : "The letter P.",
    "alpha_q"  : "The letter Q.",
    "alpha_r"  : "The letter R.",
    "alpha_s"  : "The letter S.",
    "alpha_t"  : "The letter T.",
    "alpha_u"  : "The letter U.",
    "alpha_w"  : "The letter W.",
    "alpha_x"  : "The letter X.",
    "alpha_y"  : "The letter Y.",
    "alpha_z"  : "The letter Z.",

    # ── Digits ────────────────────────────────────────────────────────────────
    "digit_0"  : "The number zero.",
    "digit_1"  : "The number one.",
    "digit_2"  : "The number two.",
    "digit_3"  : "The number three.",
    "digit_4"  : "The number four.",
    "digit_5"  : "The number five.",
    "digit_6"  : "The number six.",
    "digit_7"  : "The number seven.",
    "digit_8"  : "The number eight.",
    "digit_9"  : "The number nine.",
}

def get_sentence(sign_label):
    label = sign_label.strip().lower()
    if label in SIGN_SENTENCES:
        return SIGN_SENTENCES[label]
    # Fallback
    return label.replace("_", " ").title() + "."


# =============================================================================
# LOAD MODEL
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


# =============================================================================
# UI DRAWING
# =============================================================================
def draw_ui(frame, predicted_text, confidence, sentence, buffer_pct, state_label, state_color):
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "ISL Sign -> Text", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Q:Quit  C:Clear  SPACE:Skip | {state_label}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 1)

    # Progress bar
    bar_w = int(w * buffer_pct)
    cv2.rectangle(frame, (0, 70), (w, 80), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, 70), (bar_w, 80), state_color, -1)

    # Detected sign label
    if predicted_text:
        conf_color = (0, 255, 100) if confidence >= 0.75 else (0, 200, 255)
        cv2.putText(frame, f"Sign: {predicted_text}  ({confidence*100:.0f}%)",
                    (10, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.85, conf_color, 2)

    # Sentence bottom bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 80), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

    display = sentence if sentence else "Show a sign to begin..."
    if len(display) > 55:
        display = display[:52] + "..."

    cv2.putText(frame, display, (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return frame


# =============================================================================
# MAIN
# =============================================================================
def main():
    model, classes, feat_min, feat_rng = load_model_and_meta()

    # Download hand landmarker if needed
    model_task = os.path.join(MODEL_DIR, "hand_landmarker.task")
    if not os.path.exists(model_task):
        print("[INFO] Downloading hand landmarker model ...")
        import urllib.request
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        urllib.request.urlretrieve(url, model_task)

    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_task),
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

    STATE_COLLECTING = "COLLECTING"
    STATE_COOLDOWN   = "COOLDOWN"

    frame_buffer     = deque(maxlen=TIMESTEPS)
    predicted        = ""
    confidence       = 0.0
    current_sentence = ""
    state            = STATE_COLLECTING
    cooldown_counter = 0

    print("\n[INFO] Ready! Show a sign and hold it steady.")
    print("[INFO] Q = Quit | C = Clear | SPACE = Skip cooldown\n")

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

            # ── COLLECTING ────────────────────────────────────────────────────
            if state == STATE_COLLECTING:
                if hand_present:
                    frame_buffer.append(vec)
                else:
                    frame_buffer.clear()

                buffer_pct = len(frame_buffer) / TIMESTEPS

                if len(frame_buffer) == TIMESTEPS:
                    sequence = normalize(np.array(frame_buffer), feat_min, feat_rng)
                    X_in     = sequence[np.newaxis, ...]
                    probs    = model.predict(X_in, verbose=0)[0]
                    top_idx  = int(np.argmax(probs))
                    top_conf = float(probs[top_idx])

                    if top_conf >= CONFIDENCE_THRESHOLD:
                        predicted        = classes[top_idx]
                        confidence       = top_conf
                        current_sentence = get_sentence(predicted)
                        print(f"[PRED] {predicted}  ({top_conf*100:.1f}%)")
                        print(f"[TEXT] {current_sentence}")
                    else:
                        predicted  = ""
                        confidence = 0.0
                        print(f"[SKIP] Low confidence ({top_conf*100:.1f}%)")

                    frame_buffer.clear()
                    cooldown_counter = AUTO_RESET_FRAMES
                    state = STATE_COOLDOWN

                state_label = "COLLECTING"
                state_color = (0, 220, 80)

            # ── COOLDOWN ──────────────────────────────────────────────────────
            else:
                cooldown_counter -= 1
                buffer_pct  = 1.0 - (cooldown_counter / AUTO_RESET_FRAMES)
                state_label = f"NEXT SIGN IN {cooldown_counter}"
                state_color = (0, 140, 255)

                if cooldown_counter <= 0:
                    state = STATE_COLLECTING
                    frame_buffer.clear()
                    print("[INFO] Ready for next sign")

            # ── Draw & show ───────────────────────────────────────────────────
            frame = draw_ui(frame, predicted, confidence,
                            current_sentence, buffer_pct,
                            state_label, state_color)
            cv2.imshow("ISL Sign Language -> Text", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                predicted        = ""
                confidence       = 0.0
                current_sentence = ""
                state            = STATE_COLLECTING
                cooldown_counter = 0
                frame_buffer.clear()
            elif key == ord(' '):
                if state == STATE_COOLDOWN:
                    state            = STATE_COLLECTING
                    cooldown_counter = 0
                    frame_buffer.clear()
                    print("[INFO] Cooldown skipped!")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
