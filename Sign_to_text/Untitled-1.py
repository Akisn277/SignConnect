"""
=============================================================================
STEP 1 — Data Preparation: ISL Dataset ZIP → MediaPipe Landmarks → .npy
=============================================================================
Handles the nested ZIP structure:
    Dataset for everyday phrases and words in Indian sign language.zip
        └── images for phrases.zip
                └── images for phrases/
                        ├── again/1.png, 2.png ...
                        ├── agree/1.png, 2.png ...
                        ├── hello/1.png ...
                        └── ... (44 gestures total)

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
# CONFIG  ← only change OUTER_ZIP if your file is in a different location
# =============================================================================
OUTER_ZIP  = r"C:\Users\vidhy\Downloads\Dataset for everyday phrases and words in Indian sign language.zip"
INNER_ZIP  = "Dataset for everyday phrases and words in Indian sign language/images for phrases.zip"
IMG_PREFIX = "images for phrases/"   # prefix inside inner zip
OUTPUT_DIR = r"C:\Users\vidhy\OneDrive - SRM Institute of Science & Technology\Desktop\signconnect"

TIMESTEPS  = 10    # frames per sequence
STEP_SIZE  = 1     # smaller = more sequences (good for small datasets)

# Leave empty to use ALL gestures found in the ZIP
# Or specify a subset: GESTURES = ["hello", "yes", "no", "please", "stop"]
GESTURES   = []
# =============================================================================


def zero_vector():
    return np.zeros(42, dtype=np.float32)


def build_sequences(frames, timesteps, step):
    """Sliding window over landmark frames → list of sequences."""
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, "hand_landmarker.task")
    download_model(model_path)

    # ── MediaPipe 0.10+ Tasks API ─────────────────────────────────────────────
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

    # ── Open nested ZIPs ──────────────────────────────────────────────────────
    print("[INFO] Opening dataset ZIP …")
    outer      = zipfile.ZipFile(OUTER_ZIP)
    inner_data = outer.read(INNER_ZIP)
    inner      = zipfile.ZipFile(io.BytesIO(inner_data))

    # ── Discover all gesture folders ──────────────────────────────────────────
    all_names = inner.namelist()
    found_gestures = sorted(set(
        f[len(IMG_PREFIX):].split("/")[0]
        for f in all_names
        if f.startswith(IMG_PREFIX) and f != IMG_PREFIX
        and "/" in f[len(IMG_PREFIX):]
        and not f.endswith(IMG_PREFIX)
    ))
    found_gestures = [g for g in found_gestures if g]  # remove empty strings

    gestures = GESTURES if GESTURES else found_gestures
    print(f"[INFO] Found {len(found_gestures)} gestures in dataset.")
    print(f"[INFO] Processing {len(gestures)} gestures: {gestures}\n")

    # ── Build image lookup: gesture → sorted list of zip paths ───────────────
    gesture_files = {g: [] for g in gestures}
    for name in all_names:
        if not name.startswith(IMG_PREFIX) or name.endswith("/"):
            continue
        parts = name[len(IMG_PREFIX):].split("/")
        if len(parts) == 2:
            gesture, fname = parts
            if gesture in gesture_files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in {".png", ".jpg", ".jpeg", ".bmp"}:
                    gesture_files[gesture].append(name)

    # Sort numerically by filename (1.png, 2.png … 40.png)
    for g in gestures:
        gesture_files[g].sort(
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            if os.path.splitext(os.path.basename(p))[0].isdigit() else 0
        )

    print(f"[INFO] Images per gesture:")
    for g in gestures:
        print(f"       {g:>20} : {len(gesture_files[g]):>4} images")

    # ── Extract landmarks + build sequences ───────────────────────────────────
    all_X, all_y = [], []
    no_hand_total = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        for idx, gesture in enumerate(gestures):
            files = gesture_files[gesture]
            if not files:
                print(f"[WARN] No images for '{gesture}' — skipping.")
                continue

            print(f"\n[{idx+1}/{len(gestures)}] Processing '{gesture}' ({len(files)} images) …")
            landmark_frames = []
            no_hand_count   = 0

            for zip_path in files:
                # Read image bytes directly from ZIP (no extraction to disk)
                img_bytes = inner.read(zip_path)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img_bgr is None:
                    continue

                img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result   = landmarker.detect(mp_image)

                if result.hand_landmarks:
                    lms = result.hand_landmarks[0]
                    vec = np.array(
                        [coord for lm in lms for coord in (lm.x, lm.y)],
                        dtype=np.float32
                    )
                else:
                    vec = zero_vector()
                    no_hand_count += 1

                landmark_frames.append(vec)

            seqs = build_sequences(landmark_frames, TIMESTEPS, STEP_SIZE)
            all_X.extend(seqs)
            all_y.extend([gesture] * len(seqs))
            no_hand_total += no_hand_count
            print(f"       → {len(seqs)} sequences  |  no-hand frames: {no_hand_count}")

    inner.close()
    outer.close()

    # ── Save ──────────────────────────────────────────────────────────────────
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y)

    x_path = os.path.join(OUTPUT_DIR, "X.npy")
    y_path = os.path.join(OUTPUT_DIR, "y.npy")
    np.save(x_path, X)
    np.save(y_path, y)

    print(f"\n{'='*60}")
    print(f"  Dataset saved!")
    print(f"  X.npy → shape {X.shape}  ({X.nbytes / 1e6:.2f} MB)")
    print(f"  y.npy → shape {y.shape}")
    print(f"\n  Sequences per gesture:")
    for g in gestures:
        count = int((y == g).sum())
        print(f"    {g:>20} : {count}")
    print(f"\n  Total no-hand frames (filled with zeros): {no_hand_total}")
    print(f"{'='*60}")
    print(f"\n  ✅ Now run:  python step2_train.py")


if __name__ == "__main__":
    main()