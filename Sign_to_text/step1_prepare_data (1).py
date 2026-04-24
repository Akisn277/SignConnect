"""
=============================================================================
STEP 1 — Data Preparation: TWO Datasets → MediaPipe Landmarks → .npy
=============================================================================
Dataset 1 (Phrases):
    Already extracted folder in signconnect directory:
    Dataset for everyday phrases and words in Indian sign language/
        └── images for phrases/
                ├── again/1.png ...  (44 phrases)

Dataset 2 (Alphabets + Digits):
    archive (1).zip  (in Downloads)
        └── 0/1.png, 1/1.png ... a/1.png, b/1.png ... (33 classes)

Both datasets are combined into a single X.npy and y.npy.

Output:
    X.npy  →  shape (N, 20, 42)
    y.npy  →  shape (N,)  string labels

Run:
    python step1_prepare_data.py
=============================================================================
"""

import os
import cv2
import numpy as np
import zipfile
import urllib.request
import mediapipe as mp

# =============================================================================
# CONFIG
# =============================================================================

# Dataset 1 — ISL Phrases (already extracted folder in signconnect)
PHRASES_FOLDER = r"C:\Users\vidhy\OneDrive - SRM Institute of Science & Technology\Desktop\signconnect\Dataset for everyday phrases and words in Indian sign language\images for phrases\images for phrases"
# Dataset 2 — ISL Alphabets + Digits (zip in Downloads)
ALPHA_ZIP = r"C:\Users\vidhy\Downloads\archive (1).zip"

OUTPUT_DIR = r"C:\Users\vidhy\OneDrive - SRM Institute of Science & Technology\Desktop\signconnect"

# FIX 1: Increased from 10 → 20 for better gesture sequence learning
TIMESTEPS = 20
STEP_SIZE = 1

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
# =============================================================================


def zero_vector():
    return np.zeros(42, dtype=np.float32)


# FIX 2: Normalize landmarks relative to wrist + scale-invariant
def normalize_landmarks(vec):
    """
    Make landmarks position-independent and scale-invariant.
    - Subtracts wrist (landmark 0) so all coords are relative to wrist.
    - Divides by hand size (wrist to middle fingertip) so hand distance
      from camera does not affect predictions.
    """
    if np.all(vec == 0):
        return vec

    normalized = vec.copy()

    # Wrist is landmark 0: x=vec[0], y=vec[1]
    wrist_x = normalized[0]
    wrist_y = normalized[1]

    # Make all coords relative to wrist
    normalized[0::2] -= wrist_x   # subtract wrist_x from all x coords
    normalized[1::2] -= wrist_y   # subtract wrist_y from all y coords

    # Scale by hand size: wrist (idx 0) to middle finger tip (idx 12)
    # After subtracting wrist, landmark 12 is at normalized[24], normalized[25]
    hand_size = np.sqrt(normalized[24] ** 2 + normalized[25] ** 2)
    if hand_size > 0:
        normalized /= hand_size

    return normalized.astype(np.float32)


def build_sequences(frames, timesteps, step, augment=False):
    """
    Build fixed-length sequences from a list of landmark frames.
    FIX 3: Added augment flag — adds noise variants to increase dataset size.
    """
    n = len(frames)
    if n == 0:
        return []
    if n < timesteps:
        padded = frames + [zero_vector()] * (timesteps - n)
        return [np.stack(padded)]
    seqs = []
    for start in range(0, n - timesteps + 1, step):
        seq = np.stack(frames[start: start + timesteps])
        seqs.append(seq)

        if augment:
            # Variant 1: small random noise (simulates natural hand tremor)
            noise = np.random.normal(0, 0.02, seq.shape).astype(np.float32)
            seqs.append(seq + noise)

            # Variant 2: slightly larger noise (simulates different person)
            noise2 = np.random.normal(0, 0.04, seq.shape).astype(np.float32)
            seqs.append(seq + noise2)

    return seqs


def download_model(model_path):
    if not os.path.exists(model_path):
        print("[INFO] Downloading MediaPipe hand landmarker model (~25MB) ...")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        urllib.request.urlretrieve(url, model_path)
        print("[INFO] Model downloaded")


def extract_landmarks_from_file(landmarker, img_path):
    """Read image from disk and extract normalized hand landmark vector."""
    try:
        img_bgr = cv2.imread(img_path)
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
            return normalize_landmarks(vec), True
        return zero_vector(), False
    except Exception as e:
        print(f"[WARN] Error reading {img_path}: {e}")
        return zero_vector(), False


def extract_landmarks_from_zip(landmarker, zip_file, zip_path):
    """Read image from ZIP and extract normalized hand landmark vector."""
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
            return normalize_landmarks(vec), True
        return zero_vector(), False
    except Exception:
        return zero_vector(), False


def process_gesture_from_folder(landmarker, folder_path, gesture, idx, total, augment=False):
    """Process all images in a folder for one gesture."""
    files = sorted([
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ], key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else 0)

    print(f"\n[{idx}/{total}] '{gesture}' ({len(files)} images, augment={augment}) ...")
    frames        = []
    no_hand_count = 0

    for fname in files:
        img_path = os.path.join(folder_path, fname)
        vec, found = extract_landmarks_from_file(landmarker, img_path)
        if not found:
            no_hand_count += 1
        frames.append(vec)

    seqs = build_sequences(frames, TIMESTEPS, STEP_SIZE, augment=augment)
    print(f"       -> {len(seqs)} sequences  |  no-hand: {no_hand_count}")
    return seqs


def process_gesture_from_zip(landmarker, zip_file, files, gesture, idx, total, augment=False):
    """Process all images from a ZIP for one gesture."""
    print(f"\n[{idx}/{total}] '{gesture}' ({len(files)} images, augment={augment}) ...")
    frames        = []
    no_hand_count = 0

    for zip_path in files:
        vec, found = extract_landmarks_from_zip(landmarker, zip_file, zip_path)
        if not found:
            no_hand_count += 1
        frames.append(vec)

    seqs = build_sequences(frames, TIMESTEPS, STEP_SIZE, augment=augment)
    print(f"       -> {len(seqs)} sequences  |  no-hand: {no_hand_count}")
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

        # =====================================================================
        # DATASET 1 — ISL Phrases (extracted folder on Desktop)
        # augment=True because this dataset has only ~33 images per gesture
        # =====================================================================
        print("\n" + "="*60)
        print("  DATASET 1 — ISL Phrases (from folder)")
        print("="*60)

        if not os.path.exists(PHRASES_FOLDER):
            print(f"[ERROR] Phrases folder not found:\n        {PHRASES_FOLDER}")
            print("[INFO]  Check the folder name inside your signconnect directory.")
        else:
            gestures_1 = sorted([
                g for g in os.listdir(PHRASES_FOLDER)
                if os.path.isdir(os.path.join(PHRASES_FOLDER, g))
            ])
            print(f"[INFO] Found {len(gestures_1)} phrase gestures: {gestures_1}")

            for idx, gesture in enumerate(gestures_1):
                gesture_folder = os.path.join(PHRASES_FOLDER, gesture)
                seqs = process_gesture_from_folder(
                    landmarker, gesture_folder, gesture,
                    idx+1, len(gestures_1), augment=True
                )
                all_X.extend(seqs)
                all_y.extend([gesture] * len(seqs))

        # =====================================================================
        # DATASET 2 — ISL Alphabets + Digits (ZIP in Downloads)
        # augment=False — this dataset already has up to 300 images
        # =====================================================================
        print("\n" + "="*60)
        print("  DATASET 2 — ISL Alphabets + Digits (from ZIP)")
        print("="*60)

        if not os.path.exists(ALPHA_ZIP):
            print(f"[ERROR] Alpha ZIP not found:\n        {ALPHA_ZIP}")
            print("[INFO]  Check the path and update ALPHA_ZIP in config.")
        else:
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

            for g in gestures_2:
                gesture_files_2[g].sort(
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                    if os.path.splitext(os.path.basename(p))[0].isdigit() else 0
                )
                # Limit to 300 images max to stay balanced with dataset 1
                gesture_files_2[g] = gesture_files_2[g][:300]

            for idx, gesture in enumerate(gestures_2):
                label = f"digit_{gesture}" if gesture.isdigit() else f"alpha_{gesture}"
                seqs  = process_gesture_from_zip(
                    landmarker, alpha_zip,
                    gesture_files_2[gesture],
                    label, idx+1, len(gestures_2),
                    augment=False
                )
                all_X.extend(seqs)
                all_y.extend([label] * len(seqs))

            alpha_zip.close()

    # ── Save combined dataset ─────────────────────────────────────────────────
    if not all_X:
        print("\n[ERROR] No data was collected! Check your dataset paths above.")
        return

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y)

    x_path = os.path.join(OUTPUT_DIR, "X.npy")
    y_path = os.path.join(OUTPUT_DIR, "y.npy")
    np.save(x_path, X)
    np.save(y_path, y)

    unique, counts = np.unique(y, return_counts=True)
    print(f"\n{'='*60}")
    print(f"  Combined Dataset saved!")
    print(f"  X.npy -> shape {X.shape}  ({X.nbytes / 1e6:.2f} MB)")
    print(f"  y.npy -> shape {y.shape}")
    print(f"  Total classes : {len(unique)}")
    print(f"\n  Sequences per gesture:")
    for g, c in zip(unique, counts):
        print(f"    {g:>25} : {c}")
    print(f"{'='*60}")
    print(f"\n  Done! Now run:  python step2_train.py")


if __name__ == "__main__":
    main()
