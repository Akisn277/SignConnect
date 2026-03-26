import cv2
import numpy as np
import mediapipe as mp
import os

# ================= CONFIG =================
LABEL = "A"   # CHANGE THIS: A / B / C / D
SAVE_DIR = f"dataset/{LABEL}"
os.makedirs(SAVE_DIR, exist_ok=True)

SEQUENCE_LENGTH = 30
MAX_SAMPLES = 300
# ==========================================

# MediaPipe setup (FIXED)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Camera not found!")
    exit()

sequence = []
count = 0

print("\n[INFO] Starting collection for:", LABEL)
print("[INFO] Press Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame not received")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    # Default empty vector
    vec = np.zeros(63, dtype=np.float32)
    hand_detected = False

    if result.multi_hand_landmarks:
        hand_detected = True
        lm = result.multi_hand_landmarks[0]

        vec = np.array(
            [(p.x, p.y, p.z) for p in lm.landmark],
            dtype=np.float32
        ).flatten()

        # Draw landmarks
        for p in lm.landmark:
            x, y = int(p.x * frame.shape[1]), int(p.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # ================= BUFFER =================
    sequence.append(vec)

    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    # ================= UI =================
    status = f"Label: {LABEL} | Samples: {count}/{MAX_SAMPLES}"
    cv2.putText(frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if hand_detected:
        cv2.putText(frame, "HAND DETECTED", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NO HAND", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"Buffer: {len(sequence)}/{SEQUENCE_LENGTH}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # ================= AUTO SAVE =================
    if len(sequence) == SEQUENCE_LENGTH and hand_detected:

        np.save(f"{SAVE_DIR}/{count}.npy", np.array(sequence))
        print(f"[SAVED] {LABEL} sample {count}")

        count += 1
        sequence = []

        if count >= MAX_SAMPLES:
            print("[INFO] Dataset collection complete!")
            break

    cv2.imshow("DATA COLLECTION (A-D)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Done!")