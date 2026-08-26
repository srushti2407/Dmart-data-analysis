import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Sign Language App", layout="centered")

st.title("🤟 AI Sign Language Web App")

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
@st.cache_resource
def load_all():
    model = load_model("model.h5")
    mean = np.load("scaler_mean.npy")
    scale = np.load("scaler_scale.npy")
    labels = np.load("labels.npy", allow_pickle=True)
    return model, mean, scale, labels

model, mean, scale, labels = load_all()

# -----------------------------
# SELECT MODE
# -----------------------------
mode = st.radio("Choose Mode", ["Hand Recognition", "Text to Sign"])

# =============================
# 🤟 HAND RECOGNITION
# =============================
if mode == "Hand Recognition":

    st.info("Click Start Camera")

    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not working")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)

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
                        letter = labels[class_id]

                        cv2.putText(frame, f"{letter}", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (0, 255, 0), 3)

            FRAME_WINDOW.image(frame, channels="BGR")

        cap.release()

# =============================
# ⌨ TEXT → SIGN (VOICE ALT)
# =============================
elif mode == "Text to Sign":

    st.info("Type text (Voice alternative for web)")

    text = st.text_input("Enter text")

    if text:
        text = text.upper()
        st.success(f"Input: {text}")

        # Display letters one by one
        for char in text:
            st.write(char)