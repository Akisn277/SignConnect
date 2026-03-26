import cv2
import numpy as np
import mediapipe as mp
import os

LABEL = "A"   # change to B, C, D
SAVE_PATH = f"dataset/{LABEL}"
os.makedirs(SAVE_PATH, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

cap = cv2.VideoCapture(0)

count = 0
sequence = []

print("Press S to save sample, Q to quit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    vec = np.zeros(63)

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        vec = np.array([(p.x, p.y, p.z) for p in lm.landmark]).flatten()

    sequence.append(vec)

    if len(sequence) == 30:
        cv2.putText(frame, "Press S to save", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            np.save(f"{SAVE_PATH}/{count}.npy", np.array(sequence))
            print("Saved:", count)
            count += 1
            sequence = []

    cv2.imshow("Collect Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()