import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import time
from collections import deque, Counter

# Load model
model = tf.keras.models.load_model(r"D:\ai\asl_landmark_model_A_Z.h5")

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Voice engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Variables
history = deque(maxlen=10)
last_letter = ""
last_spoken_time = 0

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    letter = "Detecting..."

    if not result.multi_hand_landmarks:
        history.clear()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            data = []

            wrist = hand_landmarks.landmark[0]

            for lm in hand_landmarks.landmark:
                data.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])

            data = np.array(data).reshape(1, 63)

            pred = model.predict(data, verbose=0)[0]

            conf = np.max(pred)
            class_id = np.argmax(pred)

            if conf > 0.80:
                history.append(labels[class_id])

            if len(history) > 0:
                stable_letter = Counter(history).most_common(1)[0][0]
                letter = stable_letter

                current_time = time.time()

                if stable_letter != last_letter or (current_time - last_spoken_time > 2):

                    engine.say(stable_letter)
                    engine.runAndWait()

                    last_letter = stable_letter
                    last_spoken_time = current_time

            cv2.putText(
                frame,
                f"{letter} ({conf:.2f})",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

    cv2.imshow("Sign Interpreter", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()