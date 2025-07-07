import argparse
from tqdm import tqdm
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from datasets.oxford_pets import OxfordPetsSegmentation
from utils.metrics import pixel_accuracy_evaluate, compute_iou_evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Calculate GrabCut metrics on OxfordPets val split"
    )
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/classical_metrics")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    image_files = sorted(
        [f for f in os.listdir("outputs/classical") if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    preds_masks = []

    for file in image_files:
        img_path = os.path.join("outputs/classical", file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        bin_mask = (img > 0).astype(np.uint8)
        preds_masks.append(bin_mask)

    dataset = OxfordPetsSegmentation(
        root=args.dataset_root, split="val", image_size=128
    )

    classical_acuracy = []
    classical_iou = []

    if args.all:
        for i in tqdm(range(len(dataset)), desc="GrabCut"):
            _, mask_t = dataset[i]
            mask = mask_t.numpy().squeeze().astype(bool)
            iou = compute_iou_evaluate(preds_masks[i], mask)
            accuracy = pixel_accuracy_evaluate(preds_masks[i], mask)
            classical_acuracy.append(accuracy)
            classical_iou.append(iou)

    else:
        _, mask_t = dataset[args.index]
        mask = mask_t.numpy().squeeze().astype(bool)
        iou = compute_iou_evaluate(preds_masks[args.index], mask)
        accuracy = pixel_accuracy_evaluate(preds_masks[args.index], mask)
        classical_acuracy.append(accuracy)
        classical_iou.append(iou)

    os.makedirs(args.output_dir, exist_ok=True)

    classical_acuracy = np.array(classical_acuracy)
    classical_iou = np.array(classical_iou)

    acc_mean = np.mean(classical_acuracy)
    acc_std = np.std(classical_acuracy)
    iou_mean = np.mean(classical_iou)
    iou_std = np.std(classical_iou)

    print(f"Accuracy - Média: {acc_mean:.4f}, Desvio Padrão: {acc_std:.4f}")
    print(f"IoU - Média: {iou_mean:.4f}, Desvio Padrão: {iou_std:.4f}")

    plt.figure()
    plt.hist(classical_acuracy, bins=20, color="blue", edgecolor="black")
    plt.title("Histogram of Pixel Accuracy")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "histogram_accuracy.png"))
    plt.close()

    # Histogram IoU
    plt.figure()
    plt.hist(classical_iou, bins=20, color="red", edgecolor="black")
    plt.title("Histogram of IoU")
    plt.xlabel("IoU")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "histogram_iou.png"))
    plt.close()


if __name__ == "__main__":
    main()
