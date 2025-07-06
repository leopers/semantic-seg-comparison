import os
import sys
import argparse
import numpy as np
import cv2

from datasets.oxford_pets import OxfordPetsSegmentation

# Grabcut Pipeline


def get_bbox_from_mask(mask: np.ndarray, pad: int = 10) -> tuple:
    """Compute a padded bounding box from a binary GT mask."""
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h, w = mask.shape
    x0b = max(x0 - pad, 0)
    y0b = max(y0 - pad, 0)
    x1b = min(x1 + pad, w - 1)
    y1b = min(y1 + pad, h - 1)
    return (x0b, y0b, x1b - x0b, y1b - y0b)


def run_grabcut(img_bgr: np.ndarray, rect: tuple, iterations: int = 5) -> np.ndarray:
    """Run OpenCV GrabCut using a bounding box."""
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img_bgr, mask, rect, bgd, fgd, iterCount=iterations, mode=cv2.GC_INIT_WITH_RECT
    )

    result = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    return result.astype(np.uint8)


def clean_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply morphological open+close to reduce noise."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return m


def segment_with_grabcut(
    img_bgr: np.ndarray, mask_gt: np.ndarray, pad: int = 10
) -> np.ndarray:
    """Full pipeline: bounding box → grabcut → cleanup."""
    rect = get_bbox_from_mask(mask_gt, pad)
    raw = run_grabcut(img_bgr, rect)
    return clean_mask(raw)


# Scripting


def save_mask(mask: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, (mask * 255).astype(np.uint8))


def process_sample(
    index: int, dataset: OxfordPetsSegmentation, output_dir: str, pad: int
):
    img_t, mask_t = dataset[index]
    img = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask_gt = mask_t.numpy().squeeze().astype(bool)

    mask_pred = segment_with_grabcut(img_bgr, mask_gt, pad=pad)

    out_path = os.path.join(output_dir, f"mask_{index:04d}.png")
    save_mask(mask_pred, out_path)
    print(f"[{index:04d}] saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run GrabCut on OxfordPets val split")
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/classical")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--pad", type=int, default=10)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    ds = OxfordPetsSegmentation(
        root=args.dataset_root, split="val", image_size=args.image_size
    )

    if args.all:
        for i in range(len(ds)):
            process_sample(i, ds, args.output_dir, args.pad)
    else:
        process_sample(args.index, ds, args.output_dir, args.pad)


if __name__ == "__main__":
    main()
