import os

label = input("Enter label (A-Z): ")
save_path = f"dataset/{label}"

# Create folder if not exists
os.makedirs(save_path, exist_ok=True)

import cv2
import mediapipe as mp
import os

label = input("Enter label (A-Z): ").upper()
save_path = f"dataset/{label}"

os.makedirs(save_path, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
count = 0

print("Press S to save image")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Saved: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        file_name = f"{save_path}/{count}.jpg"
        cv2.imwrite(file_name, frame)
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()