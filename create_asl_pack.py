import os
import shutil

source = "D:/ai/dataset"
dest = "D:/ai/asl_images"

os.makedirs(dest, exist_ok=True)

for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    folder = os.path.join(source, f"{letter}-samples")

    if os.path.exists(folder):
        files = os.listdir(folder)
        if files:
            src_path = os.path.join(folder, files[0])
            dst_path = os.path.join(dest, f"{letter}.jpg")

            shutil.copy(src_path, dst_path)
            print(f"Copied {letter}")

print("✅ Done!")