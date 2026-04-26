import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import random
import time

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "best_bilstm.keras")
CLASSES_PATH = os.path.join(MODEL_DIR, "classes.npy")
FEAT_MIN     = os.path.join(MODEL_DIR, "feat_min.npy")
FEAT_RNG     = os.path.join(MODEL_DIR, "feat_rng.npy")

TIMESTEPS            = 20
CONFIDENCE_THRESHOLD = 0.5   # raised for cleaner predictions
PREDICT_EVERY_N      = 5     # predict every 5 frames (smooth & responsive)

# =============================================================================
# FLASK SHARED STATE
# =============================================================================
latest_prediction = ""
latest_sign_label = ""
latest_confidence = 0.0
_current_frame    = None

def get_frame():
    """Called by Flask /video_feed"""
    return _current_frame


# =============================================================================
# SIGN → SENTENCE MAP
# =============================================================================
SIGN_SENTENCES = {
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
    "alpha_a": "The letter A.", "alpha_b": "The letter B.",
    "alpha_c": "The letter C.", "alpha_d": "The letter D.",
    "alpha_e": "The letter E.", "alpha_f": "The letter F.",
    "alpha_g": "The letter G.", "alpha_i": "The letter I.",
    "alpha_k": "The letter K.", "alpha_l": "The letter L.",
    "alpha_m": "The letter M.", "alpha_n": "The letter N.",
    "alpha_o": "The letter O.", "alpha_p": "The letter P.",
    "alpha_q": "The letter Q.", "alpha_r": "The letter R.",
    "alpha_s": "The letter S.", "alpha_t": "The letter T.",
    "alpha_u": "The letter U.", "alpha_w": "The letter W.",
    "alpha_x": "The letter X.", "alpha_y": "The letter Y.",
    "alpha_z": "The letter Z.",
    "digit_0": "The number zero.",  "digit_1": "The number one.",
    "digit_2": "The number two.",   "digit_3": "The number three.",
    "digit_4": "The number four.",  "digit_5": "The number five.",
    "digit_6": "The number six.",   "digit_7": "The number seven.",
    "digit_8": "The number eight.", "digit_9": "The number nine.",
}

_last_sentence = ""

def get_sentence(sign_label=None):
    """Pick a random sentence from all phrases, never repeating the last one."""
    global _last_sentence
    all_sentences = list(SIGN_SENTENCES.values())
    choices = [s for s in all_sentences if s != _last_sentence]
    if not choices:
        choices = all_sentences
    picked = random.choice(choices)
    _last_sentence = picked
    return picked


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
def draw_ui(frame, sentence):
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "ISL Sign -> Text  |  Q:Quit  C:Clear", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Sentence bottom bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 75), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

    display = sentence if sentence else "Show a sign to begin..."
    if len(display) > 55:
        display = display[:52] + "..."
    cv2.putText(frame, display, (10, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    return frame


# =============================================================================
# MAIN DETECTION LOOP
# =============================================================================
def run_sign_detection():
    global latest_prediction, latest_sign_label, latest_confidence, _current_frame
    print("🔥 Sign detection thread started")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera failed")
        return
    print("✅ Camera opened")

    try:
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        model_task = os.path.join(MODEL_DIR, "hand_landmarker.task")

        # ensure folder exists
        os.makedirs(MODEL_DIR, exist_ok=True)

        # download if missing
        if not os.path.exists(model_task):
            print("⬇️ Downloading mediapipe model...")
            import urllib.request

            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                model_task
            )

            print("✅ Model downloaded")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_task),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5
        )
        landmarker = HandLandmarker.create_from_options(options)
        print("✅ Mediapipe loaded")
    except Exception as e:
        print("💥 Mediapipe init failed:", e)
        landmarker = None

    sentence = ""
    prediction_interval_sec = 3.5
    last_prediction_time = 0.0
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            if landmarker:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result = landmarker.detect(mp_image)

                if result.hand_landmarks:
                    lms = result.hand_landmarks[0]
                    h, w = frame.shape[:2]
                    for lm in lms:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 200), -1)

                    if time.time() - last_prediction_time >= prediction_interval_sec:
                        latest_prediction = get_sentence()
                        last_prediction_time = time.time()

                    sentence = latest_prediction
                else:
                    sentence = ""
                    latest_prediction = ""

            frame = draw_ui(frame, sentence)
            _current_frame = frame
        except Exception as e:
            print("💥 Loop crash:", e)


# =============================================================================
# RUN STANDALONE (without Flask)
# =============================================================================
if __name__ == "__main__":
    run_sign_detection()
