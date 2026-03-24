

"""
=============================================================================
STEP 1 — Data Preparation: TWO Datasets → MediaPipe Landmarks → .npy
=============================================================================
Dataset 1 (Phrases):
    Dataset for everyday phrases and words in Indian sign language.zip
        └── images for phrases.zip
                └── images for phrases/
                        ├── again/1.png ...  (44 phrases)

Dataset 2 (Alphabets + Digits):
    archive (1).zip
        └── 0/1.png, 1/1.png ... a/1.png, b/1.png ... (33 classes)

Both datasets are combined into a single X.npy and y.npy.

Output:
    X.npy  →  shape (N, 30, 42)
    y.npy  →  shape (N,)  string labels

Run:
    python step1_prepare_data.py
=============================================================================
"""

import io
import os
import cv2
import numpy as np
import zipfile
import urllib.request
import mediapipe as mp

# =============================================================================
# CONFIG
# =============================================================================
# Dataset 1 — ISL Phrases (nested ZIP)
PHRASES_OUTER_ZIP = r"C:\Users\vidhy\Downloads\Dataset for everyday phrases and words in Indian sign language.zip"
PHRASES_INNER_ZIP = "Dataset for everyday phrases and words in Indian sign language/images for phrases.zip"
PHRASES_PREFIX    = "images for phrases/"

# Dataset 2 — ISL Alphabets + Digits (flat ZIP, folder = label)
ALPHA_ZIP         = r"C:\Users\vidhy\Downloads\archive (1).zip"

OUTPUT_DIR = r"C:\Users\vidhy\OneDrive - SRM Institute of Science & Technology\Desktop\signconnect"

TIMESTEPS  = 10
STEP_SIZE  = 1

IMG_EXTS   = {".png", ".jpg", ".jpeg", ".bmp"}
# =============================================================================


def zero_vector():
    return np.zeros(42, dtype=np.float32)


def build_sequences(frames, timesteps, step):
    n = len(frames)
    if n == 0:
        return []
    if n < timesteps:
        padded = frames + [zero_vector()] * (timesteps - n)
        return [np.stack(padded)]
    seqs = []
    for start in range(0, n - timesteps + 1, step):
        seqs.append(np.stack(frames[start: start + timesteps]))
    return seqs


def download_model(model_path):
    if not os.path.exists(model_path):
        print("[INFO] Downloading MediaPipe hand landmarker model (~25MB) …")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        urllib.request.urlretrieve(url, model_path)
        print("[INFO] Model downloaded ✅")


def extract_landmarks(landmarker, zip_file, zip_path):
    """Read image from ZIP and extract hand landmark vector."""
    try:
        img_bytes = zip_file.read(zip_path)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return zero_vector(), False
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result   = landmarker.detect(mp_image)
        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            vec = np.array(
                [coord for lm in lms for coord in (lm.x, lm.y)],
                dtype=np.float32
            )
            return vec, True
        return zero_vector(), False
    except Exception:
        return zero_vector(), False


def process_gesture(landmarker, zip_file, files, gesture, idx, total):
    """Extract landmarks for all images of one gesture → sequences."""
    print(f"\n[{idx}/{total}] '{gesture}' ({len(files)} images) …")
    frames        = []
    no_hand_count = 0

    for zip_path in files:
        vec, found = extract_landmarks(landmarker, zip_file, zip_path)
        if not found:
            no_hand_count += 1
        frames.append(vec)

    seqs = build_sequences(frames, TIMESTEPS, STEP_SIZE)
    print(f"       → {len(seqs)} sequences  |  no-hand: {no_hand_count}")
    return seqs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, "hand_landmarker.task")
    download_model(model_path)

    # ── MediaPipe setup ───────────────────────────────────────────────────────
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

    all_X, all_y = [], []

    with HandLandmarker.create_from_options(options) as landmarker:

        # ══════════════════════════════════════════════════════════════════════
        # DATASET 1 — ISL Phrases (nested ZIP)
        # ══════════════════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("  DATASET 1 — ISL Phrases")
        print("="*60)

        outer      = zipfile.ZipFile(PHRASES_OUTER_ZIP)
        inner_data = outer.read(PHRASES_INNER_ZIP)
        inner      = zipfile.ZipFile(io.BytesIO(inner_data))
        all_names  = inner.namelist()

        # Discover gestures
        gestures_1 = sorted(set(
            f[len(PHRASES_PREFIX):].split("/")[0]
            for f in all_names
            if f.startswith(PHRASES_PREFIX)
            and "/" in f[len(PHRASES_PREFIX):]
            and not f.endswith("/")
        ))
        gestures_1 = [g for g in gestures_1 if g]
        print(f"[INFO] Found {len(gestures_1)} phrase gestures")

        # Build file lists
        gesture_files_1 = {g: [] for g in gestures_1}
        for name in all_names:
            if not name.startswith(PHRASES_PREFIX) or name.endswith("/"):
                continue
            parts = name[len(PHRASES_PREFIX):].split("/")
            if len(parts) == 2:
                gesture, fname = parts
                if gesture in gesture_files_1:
                    if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                        gesture_files_1[gesture].append(name)

        for g in gestures_1:
            gesture_files_1[g].sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                if os.path.splitext(os.path.basename(p))[0].isdigit() else 0
            )

        for idx, gesture in enumerate(gestures_1):
            seqs = process_gesture(landmarker, inner,
                                   gesture_files_1[gesture],
                                   gesture, idx+1, len(gestures_1))
            all_X.extend(seqs)
            all_y.extend([gesture] * len(seqs))

        inner.close()
        outer.close()

        # ══════════════════════════════════════════════════════════════════════
        # DATASET 2 — ISL Alphabets + Digits
        # ══════════════════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("  DATASET 2 — ISL Alphabets + Digits")
        print("="*60)

        alpha_zip  = zipfile.ZipFile(ALPHA_ZIP)
        all_names2 = alpha_zip.namelist()

        gestures_2 = sorted(set(
            f.split("/")[0]
            for f in all_names2
            if "/" in f and not f.endswith("/")
        ))
        gestures_2 = [g for g in gestures_2 if g]
        print(f"[INFO] Found {len(gestures_2)} alphabet/digit gestures")

        gesture_files_2 = {g: [] for g in gestures_2}
        for name in all_names2:
            if name.endswith("/"):
                continue
            parts = name.split("/")
            if len(parts) == 2:
                folder, fname = parts
                if folder in gesture_files_2:
                    if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                        gesture_files_2[folder].append(name)

        # Sort numerically
        for g in gestures_2:
            gesture_files_2[g].sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                if os.path.splitext(os.path.basename(p))[0].isdigit() else 0
            )
            # Limit to 300 images max to keep balanced with dataset 1
            gesture_files_2[g] = gesture_files_2[g][:300]

        for idx, gesture in enumerate(gestures_2):
            # Use "digit_0", "digit_1" etc for digits to avoid confusion
            label = f"digit_{gesture}" if gesture.isdigit() else f"alpha_{gesture}"
            seqs  = process_gesture(landmarker, alpha_zip,
                                    gesture_files_2[gesture],
                                    label, idx+1, len(gestures_2))
            all_X.extend(seqs)
            all_y.extend([label] * len(seqs))

        alpha_zip.close()

    # ── Save combined dataset ─────────────────────────────────────────────────
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y)

    x_path = os.path.join(OUTPUT_DIR, "X.npy")
    y_path = os.path.join(OUTPUT_DIR, "y.npy")
    np.save(x_path, X)
    np.save(y_path, y)

    unique, counts = np.unique(y, return_counts=True)
    print(f"\n{'='*60}")
    print(f"  Combined Dataset saved!")
    print(f"  X.npy → shape {X.shape}  ({X.nbytes / 1e6:.2f} MB)")
    print(f"  y.npy → shape {y.shape}")
    print(f"  Total classes : {len(unique)}")
    print(f"\n  Sequences per gesture:")
    for g, c in zip(unique, counts):
        print(f"    {g:>25} : {c}")
    print(f"{'='*60}")
    print(f"\n  ✅ Now run:  python step2_train.py")


if __name__ == "__main__":
    main()
