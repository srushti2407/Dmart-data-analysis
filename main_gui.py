import tkinter as tk
import cv2
import numpy as np
import mediapipe as mp
import threading
import speech_recognition as sr
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import os

# ==============================
# CREATE ROOT FIRST (IMPORTANT)
# ==============================
root = tk.Tk()
root.title("AI Interpreter")
root.geometry("500x600")
root.config(bg="#f0f2f5")

# ==============================
# LOAD MODEL FILES
# ==============================
model = load_model("model.h5")
mean = np.load("scaler_mean.npy")
scale = np.load("scaler_scale.npy")
labels = np.load("labels.npy", allow_pickle=True)

# ==============================
# LOAD SIGN IMAGES (AFTER ROOT)
# ==============================
image_folder = "asl_images"
images = {}

for file in os.listdir(image_folder):
    if file.endswith(".jpg") or file.endswith(".png"):
        key = os.path.splitext(file)[0].upper()
        path = os.path.join(image_folder, file)

        img = Image.open(path)
        img = img.resize((200, 200))

        images[key] = ImageTk.PhotoImage(img)

print("Loaded images:", images.keys())

# ==============================
# GUI DESIGN (SAME AS YOUR IMAGE)
# ==============================
title = tk.Label(root, text="AI Sign & Voice Interpreter",
                 font=("Arial", 18, "bold"), bg="#f0f2f5")
title.pack(pady=20)

camera_label = tk.Label(root)
camera_label.pack()

img_label = tk.Label(root)
img_label.pack(pady=10)

status_label = tk.Label(root, text="Status: Idle",
                        font=("Arial", 12), bg="#f0f2f5")
status_label.pack(pady=10)

# ==============================
# HAND DETECTION SETUP
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = None
running = False
pred_history = []

# ==============================
# START HAND DETECTION
# ==============================
def start_hand():
    global cap, running

    if running:
        return

    cap = cv2.VideoCapture(0)
    running = True
    status_label.config(text="Status: Camera Started")

    update_frame()

def update_frame():
    global running, cap, pred_history

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    prediction_text = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            coords = []
            base = hand_landmarks.landmark[0]

            for lm in hand_landmarks.landmark:
                coords.extend([
                    lm.x - base.x,
                    lm.y - base.y,
                    lm.z - base.z
                ])

            data = np.array(coords).reshape(1, -1)

            if data.shape[1] == mean.shape[0]:
                data = (data - mean) / scale

                preds = model.predict(data, verbose=0)
                class_id = np.argmax(preds)

                # smoothing
                pred_history.append(class_id)
                if len(pred_history) > 8:
                    pred_history.pop(0)

                class_id = max(set(pred_history), key=pred_history.count)
                prediction_text = labels[class_id]

                # show sign image
                if prediction_text in images:
                    img_label.config(image=images[prediction_text])
                    img_label.image = images[prediction_text]

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    status_label.config(text=f"Detected: {prediction_text}")

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((350, 300))
    imgtk = ImageTk.PhotoImage(img)

    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    root.after(10, update_frame)

def stop_hand():
    global cap, running
    running = False
    if cap:
        cap.release()
    camera_label.config(image="")
    status_label.config(text="Status: Stopped")

# ==============================
# VOICE RECOGNITION
# ==============================
def start_voice():
    def run():
        recognizer = sr.Recognizer()

        try:
            with sr.Microphone() as source:
                status_label.config(text="Listening...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source)

            text = recognizer.recognize_google(audio).upper()
            status_label.config(text=f"Voice: {text}")

            delay = 0
            for char in text:
                if char in images:
                    root.after(delay, lambda c=char: img_label.config(image=images[c]))
                    delay += 700

        except:
            status_label.config(text="Voice Error")

    threading.Thread(target=run, daemon=True).start()

# ==============================
# BUTTONS (SAME STYLE)
# ==============================
btn_hand = tk.Button(root, text="Hand Recognition",
                     width=20, height=2,
                     command=start_hand)
btn_hand.pack(pady=10)

btn_voice = tk.Button(root, text="Voice Recognition",
                      width=20, height=2,
                      command=start_voice)
btn_voice.pack(pady=10)

btn_stop = tk.Button(root, text="Stop Camera",
                     width=20, height=2,
                     command=stop_hand)
btn_stop.pack(pady=10)

# ==============================
# RUN APP
# ==============================
root.mainloop()