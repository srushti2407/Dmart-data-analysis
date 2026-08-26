import speech_recognition as sr
import tkinter as tk
from PIL import Image, ImageTk
import os
import threading

# ======================
# GUI
# ======================
root = tk.Tk()
root.title("Voice To Sign AI")
root.geometry("500x500")

label = tk.Label(root, text="Press Start", font=("Arial", 20))
label.pack(pady=20)

img_label = tk.Label(root)
img_label.pack(pady=20)

# ======================
# LOAD IMAGES
# ======================
image_folder = "asl_images"
images = {}

if not os.path.exists(image_folder):
    print("❌ Folder not found:", image_folder)
else:
    for file in os.listdir(image_folder):
        if file.endswith(".jpg"):   # ✅ fixed

            key = os.path.splitext(file)[0].upper()
            path = os.path.join(image_folder, file)

            img = Image.open(path)
            img = img.resize((200, 200))

            images[key] = ImageTk.PhotoImage(img)

print("✅ Loaded images:", list(images.keys()))

# ======================
# SAFE UI UPDATE FUNCTION
# ======================
def show_image(img):
    img_label.config(image=img)
    img_label.image = img   # 🔥 VERY IMPORTANT

def update_text(text):
    label.config(text=text)

# ======================
# VOICE → SIGN LOGIC
# ======================
def process_voice():
    r = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            root.after(0, update_text, "Listening...")

            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source)

        text = r.recognize_google(audio).upper()
        root.after(0, update_text, f"You Said: {text}")

        words = text.split()

        delay = 0

        for word in words:

            print("Processing:", word)

            # WORD MATCH
            if word in images:
                root.after(delay, show_image, images[word])
                delay += 1000

            else:
                # LETTER FALLBACK
                for char in word:
                    if char in images:
                        root.after(delay, show_image, images[char])
                        delay += 600
                    else:
                        print("Missing:", char)

    except sr.UnknownValueError:
        root.after(0, update_text, "Could not understand")
    except sr.RequestError:
        root.after(0, update_text, "Internet error")
    except Exception as e:
        root.after(0, update_text, str(e))

# ======================
# THREAD WRAPPER
# ======================
def listen():
    threading.Thread(target=process_voice, daemon=True).start()

# ======================
# BUTTON
# ======================
btn = tk.Button(root, text="🎤 Start Voice", font=("Arial", 16), command=listen)
btn.pack(pady=20)

root.mainloop()