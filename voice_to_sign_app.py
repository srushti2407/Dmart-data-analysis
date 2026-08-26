import speech_recognition as sr
from tkinter import *
from PIL import Image, ImageTk

root = Tk()
root.title("Voice To Sign")
root.geometry("500x500")

label = Label(root, text="Speak Something", font=("Arial",18))
label.pack(pady=20)

def listen():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        label.config(text="Listening...")
        root.update()

        audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            label.config(text=text)

        except:
            label.config(text="Could not understand")

btn = Button(root,text="🎤 Start Mic",font=("Arial",14),command=listen)
btn.pack(pady=20)

root.mainloop()