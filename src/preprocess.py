# src/preprocess.py
import os
from PIL import Image
import torch
import torchvision.transforms as T

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def preprocess_image(img_path):
    transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])  # [-1,1] range
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img)

def preprocess_folder(subfolder):
    input_path = os.path.join(INPUT_DIR, subfolder)
    output_path = os.path.join(OUTPUT_DIR, subfolder)
    ensure_dir(output_path)

    for fname in os.listdir(input_path):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            img_tensor = preprocess_image(os.path.join(input_path, fname))
            torch.save(img_tensor, os.path.join(output_path, fname + ".pt"))
            print(f"Saved {fname}.pt")

def main():
    ensure_dir(OUTPUT_DIR)
    preprocess_folder("psycho")
    preprocess_folder("ragingBull")

if __name__ == "__main__":
    main()