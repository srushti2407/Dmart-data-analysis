import cv2
import mediapipe as mp
import os
import csv

DATASET_DIR = "dataset"
OUTPUT_CSV = "landmarks_A_Z.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

header = []
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]
header.append("label")

rows = []

for label in os.listdir(DATASET_DIR):
    folder = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(folder):
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            # ✅ NORMALIZATION (IMPORTANT)
            x0 = hand.landmark[0].x
            y0 = hand.landmark[0].y
            z0 = hand.landmark[0].z

            row = []
            for lm in hand.landmark:
                row.extend([
                    lm.x - x0,
                    lm.y - y0,
                    lm.z - z0
                ])

            row.append(label)
            rows.append(row)

hands.close()

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("✅ Landmark extraction complete")