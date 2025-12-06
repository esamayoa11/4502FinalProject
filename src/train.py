# src/train.py
# train.py
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import lpips  # perceptual loss
from glob import glob
from renderer import SimpleRenderer

# -----------------------------
# CONFIG
# -----------------------------
FILMS = ["psycho", "ragingBull"]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
LR = 0.0005  # reduced learning rate for stability

# -----------------------------
# HELPERS
# -----------------------------
def load_tensors(film):
    folder = os.path.join(PROCESSED_DIR, film)
    tensor_paths = glob(os.path.join(folder, "*.pt"))
    tensors = [torch.load(p).to(DEVICE) for p in tensor_paths]
    return tensors

def save_results(film, final_image, renderer, losses):
    out_dir = os.path.join(RESULTS_DIR, film)
    os.makedirs(out_dir, exist_ok=True)

    # Final rendered image
    plt.imsave(os.path.join(out_dir, "final_image.png"),
               final_image.squeeze().cpu().numpy(), cmap='gray')

    # Renderer parameters
    torch.save(renderer.state_dict(), os.path.join(out_dir, "params.pt"))

    # Loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("LPIPS Loss")
    plt.title(f"Loss Curve - {film}")
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    # Save per-step logs
    df = pd.DataFrame(losses, columns=["loss"])
    df.to_csv(os.path.join(out_dir, "logs.csv"), index_label="iteration")

# -----------------------------
# TRAINING LOOP
# -----------------------------
def train_film(film):
    print(f"\n=== Training {film} ===")
    targets = load_tensors(film)
    if len(targets) == 0:
        print(f"No preprocessed tensors found for {film}")
        return

    renderer = SimpleRenderer().to(DEVICE)
    optimizer = optim.Adam(renderer.parameters(), lr=LR, weight_decay=1e-5)

    # LPIPS perceptual loss
    loss_fn = lpips.LPIPS(net='alex').to(DEVICE)

    losses = []

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for target in targets:
            optimizer.zero_grad()
            output = renderer()  # (1,H,W)

            # Repeat grayscale to 3 channels and clamp to [-1,1] for LPIPS
            output_3c = torch.clamp(output, 0, 1).repeat(1,3,1,1) * 2 - 1
            target_3c = torch.clamp(target, 0, 1).repeat(1,3,1,1) * 2 - 1

            loss = loss_fn(output_3c, target_3c)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(targets)
        losses.append(avg_loss)

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - LPIPS Loss: {avg_loss:.6f}")

    # Save results after training
    final_image = renderer().detach()
    save_results(film, final_image, renderer, losses)
    print(f"Training finished for {film}. Results saved to {RESULTS_DIR}/{film}/")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    for film in FILMS:
        train_film(film)