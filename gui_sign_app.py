import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess
import sys
import os


# --------------------------------
# Functions
# --------------------------------
def start_hand_sign():
    try:
        status.config(text="Opening Hand Sign Recognition...", fg="blue")
        subprocess.Popen([sys.executable, "sign_interpreter.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Could not open hand_sign.py\n{e}")


def start_voice():
    try:
        status.config(text="Opening Voice Recognition...", fg="blue")
        subprocess.Popen([sys.executable, "voice_sign.py"])
    except Exception as e:
        messagebox.showerror("Error", f"Could not open voice_sign.py\n{e}")


def close_app():
    root.destroy()


# --------------------------------
# Main Window
# --------------------------------
root = tk.Tk()
root.title("AI Sign Language Translator")
root.geometry("600x650")
root.configure(bg="#f0f8ff")


# --------------------------------
# Title
# --------------------------------
title = tk.Label(
    root,
    text="🤟 AI Sign Language Translator",
    font=("Arial", 20, "bold"),
    bg="#f0f8ff",
    fg="#003366"
)
title.pack(pady=20)


# --------------------------------
# Load Image
# --------------------------------
try:
    image_path = r"D:\ai\asl_images\Y.jpg"

    if os.path.exists(image_path):
        img = Image.open(image_path)
        img = img.resize((300, 300))
        photo = ImageTk.PhotoImage(img)

        panel = tk.Label(root, image=photo, bg="#f0f8ff")
        panel.image = photo
        panel.pack(pady=10)

except:
    pass


# --------------------------------
# Status Label
# --------------------------------
status = tk.Label(
    root,
    text="Ready",
    font=("Arial", 14),
    bg="#f0f8ff",
    fg="green"
)
status.pack(pady=15)


# --------------------------------
# Buttons
# --------------------------------
btn1 = tk.Button(
    root,
    text="✋ Hand Sign Recognition",
    font=("Arial", 14),
    bg="#4CAF50",
    fg="white",
    width=25,
    command=start_hand_sign
)
btn1.pack(pady=10)


btn2 = tk.Button(
    root,
    text="🎤 Voice Recognition",
    font=("Arial", 14),
    bg="#2196F3",
    fg="white",
    width=25,
    command=start_voice
)
btn2.pack(pady=10)


btn3 = tk.Button(
    root,
    text="❌ Exit",
    font=("Arial", 14),
    bg="#f44336",
    fg="white",
    width=25,
    command=close_app
)
btn3.pack(pady=10)


# --------------------------------
# Run App
# --------------------------------
root.mainloop()