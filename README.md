# Differentiable Cinematic Rendering: Optimizing Lens and Lighting Parameters for Stylized Image Synthesis

## Project Description
This project investigates whether a minimal differentiable rendering pipeline—reduced to basic interactions between light, lens, and surface—can recover meaningful film-specific optical signatures. Using reference frames from *Psycho* and *Raging Bull*, the system optimizes lens distortion, lighting, and camera parameters using perceptual losses (LPIPS, SSIM, MSE).

## Core Questions
- What optical characteristics distinguish *Psycho* vs. *Raging Bull*?
- Can differentiable rendering recover film-specific lens behavior?
- Can perceptual losses reveal stylistic “optical DNA”?

## Summary of Answers
- Yes: even a minimal sphere-scene can reproduce historically consistent lens tendencies.
- Distortion curvature, tonal rolloff, and perceptual metrics converge toward each film’s optical identity.
- Style signatures are encoded primarily in the optical pathway—not scene complexity.
- Gradient-based inference can reveal measurable stylistic structure typically discussed only qualitatively.

## Final Project Paper
Found in project main folder 

## Project Presentation
Found in project main folder

## Repository Contents
- `src/` – Python scripts, optimization loops, renderer code
- `data/` – processed 128×128 frames
- `results/` – optimization outputs, plots
- `README.md`

## How to Run the Project
1) clone the repo
2) create & activate environent:
conda create -n filmstyle python=3.10 -y
conda activate filmstyle

if using venv:
python3 -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows

3) Install Dependencies

This project uses a minimal stack. Install the required libraries manually:

pip install torch torchvision torchaudio
pip install pyTorch3D
pip install lpips
pip install matplotlib
pip install numpy
pip install pillow

If PyTorch3D installation fails, see: https://github.com/facebookresearch/pytorch3d

4) Run the full pipeline
You can run everything manually or use the provided helper script by making the following script executable: 
chmod +x quick_run.sh

Run it: 
./quick_run.sh

This will:
- Preprocesses images
- Run training for Psycho + Raging Bull
- Save loss curves and rendered outputs into results/
