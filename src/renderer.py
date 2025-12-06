#renderer.py
import torch
import torch.nn as nn

class SimpleRenderer(nn.Module):
    """
    Minimal differentiable renderer:
    - Sphere shading
    - Single directional light
    - Radial distortion
    - Exposure control

    Improvements:
    - Clamps all learnable parameters to reasonable ranges
    - Clamps output image to [0,1] for numerical stability
    """

    def __init__(self, resolution=128):
        super().__init__()

        # Learnable parameters
        self.light_intensity = nn.Parameter(torch.tensor(1.0))
        self.camera_distance = nn.Parameter(torch.tensor(2.0))
        self.distortion_k1 = nn.Parameter(torch.tensor(0.0))
        self.exposure = nn.Parameter(torch.tensor(1.0))

        # Precompute a grid of coordinates for the image
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        self.grid_x, self.grid_y = torch.meshgrid(x, y, indexing="ij")
        self.res = resolution

    def forward(self):
        # -------------------------
        # Clamp learnable parameters to prevent NaNs
        # -------------------------
        light = torch.clamp(self.light_intensity, 0.0, 2.0)
        exposure = torch.clamp(self.exposure, 0.0, 2.0)
        k1 = torch.clamp(self.distortion_k1, -0.3, 0.3)  # small radial distortion

        # Sphere radius
        r = 0.75

        # Radial distortion
        r2 = self.grid_x**2 + self.grid_y**2
        distorted_x = self.grid_x * (1 + k1 * r2)
        distorted_y = self.grid_y * (1 + k1 * r2)

        # Mask of pixels inside the sphere
        mask = (distorted_x**2 + distorted_y**2) <= r**2

        # Surface normals
        z = torch.sqrt(torch.clamp(r**2 - distorted_x**2 - distorted_y**2, min=0.0))
        normals = torch.stack([distorted_x, distorted_y, z], dim=0)

        # Fixed directional light
        light_dir = torch.tensor([-0.5, -0.5, 1.0], device=normals.device).reshape(3,1,1)
        light_dir = light_dir / torch.norm(light_dir)

        # Lambertian shading
        dot = (normals * light_dir).sum(0).clamp(min=0.0)

        # Apply intensity and exposure
        img = dot * light * exposure

        # Apply mask and clamp to [0,1]
        img = img * mask
        img = torch.clamp(img, 0.0, 1.0)

        # Return 1-channel image
        return img.unsqueeze(0)  # (1,H,W)