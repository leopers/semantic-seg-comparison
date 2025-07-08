import os
import random
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from datasets.oxford_pets import OxfordPetsSegmentation
from models.unet import UNet

# Device
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# Load model
model = UNet().to(device)
model.load_state_dict(torch.load("models/trained/unet.pth", map_location=device))
model.eval()

# Dataset
dataset = OxfordPetsSegmentation(root="dataset", split="val", image_size=128)
indices = random.sample(range(len(dataset)), 2)


def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    image = image.copy()
    mask_rgb = np.stack([mask * c for c in color], axis=-1)
    return np.where(
        mask[..., None].astype(bool),
        (1 - alpha) * image + alpha * mask_rgb,
        image,
    ).astype(np.uint8)


# Plotting

row_titles = ["Original", "Ground Truth", "Network", "GrabCut"]
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 10))
fig.subplots_adjust(hspace=0.3, wspace=0.05, top=0.95)


for col_idx, idx in enumerate(indices):
    img_t, mask_t = dataset[idx]
    img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    mask_gt = mask_t.squeeze().numpy().astype(np.uint8)

    with torch.no_grad():
        pred = model(img_t.unsqueeze(0).to(device))
        pred_bin = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    grabcut_path = os.path.join("outputs/classical", f"{idx}.png")
    grabcut_mask = cv2.imread(grabcut_path, cv2.IMREAD_GRAYSCALE)
    grabcut_mask = (grabcut_mask > 0).astype(np.uint8)

    overlays = [
        img_np,
        overlay_mask(img_np, mask_gt, (0, 255, 0)),
        overlay_mask(img_np, pred_bin, (255, 0, 0)),
        overlay_mask(img_np, grabcut_mask, (0, 0, 255)),
    ]

    for row_idx, overlay in enumerate(overlays):
        ax = axes[row_idx, col_idx]
        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title(row_titles[row_idx], fontsize=10)

# Save result
os.makedirs("outputs/visualize", exist_ok=True)
plt.savefig("outputs/visualize/visualize.pdf", dpi=300)
plt.show()
