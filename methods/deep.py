import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils.metrics import pixel_accuracy, compute_iou
from datasets.oxford_pets import OxfordPetsSegmentation
from models.unet import UNet

# U-net pipeline


def train_model(model, train_loader, val_loader, device, args):
    """
    Train the U-Net model on the Oxford Pets dataset, using BCE loss and Adam optimizer.
    Saves the model and optionally plots training loss.
    """

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_iou": [],
        "val_iou": [],
    }

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()

        # Training metrics
        train_preds, train_targets = [], []
        with torch.no_grad():
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                train_preds.append(preds.cpu())
                train_targets.append(masks.cpu())
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_acc = pixel_accuracy(train_preds, train_targets)
        train_iou = compute_iou(train_preds, train_targets)

        # Validation metrics
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, masks).item()
                val_preds.append(preds.cpu())
                val_targets.append(masks.cpu())
        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_acc = pixel_accuracy(val_preds, val_targets)
        val_iou = compute_iou(val_preds, val_targets)

        # Save metrics
        avg_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

    # Save model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save(model.state_dict(), args.model_path)

    # Plot training if requested
    if args.plot:
        os.makedirs("outputs/training", exist_ok=True)
        epochs = range(1, args.epochs + 1)

        # Loss
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.plot(epochs, history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("outputs/training/loss_curve.png")

        # Accuracy
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_acc"], label="Train Accuracy")
        plt.plot(epochs, history["val_acc"], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Pixel Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("outputs/training/accuracy_curve.png")

        # IoU
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_iou"], label="Train IoU")
        plt.plot(epochs, history["val_iou"], label="Val IoU")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.title("IoU Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("outputs/training/iou_curve.png")


def predict(model, val_loader, device, args):
    """
    Run inference on the validation set and save predictions
    """

    model.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, _) in enumerate(tqdm(val_loader, desc="Predicting")):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.squeeze().cpu().numpy()
            bin_mask = (preds > 0.5).astype(np.uint8) * 255

            out_path = os.path.join(args.output_dir, f"{idx}.png")
            cv2.imwrite(out_path, bin_mask)


# Scripting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--model-path", type=str, default="models/trained/unet.pth")
    parser.add_argument("--output-dir", type=str, default="outputs/unet")
    parser.add_argument("--plot", action="store_true", help="Plot and save training")
    args = parser.parse_args()

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = UNet().to(device)

    if args.mode == "train":
        train_ds = OxfordPetsSegmentation(root="dataset", split="train", image_size=128)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True
        )

        val_ds = OxfordPetsSegmentation(root="dataset", split="val", image_size=128)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

        train_model(model, train_loader, val_loader, device, args)

    elif args.mode == "predict":
        val_ds = OxfordPetsSegmentation(root="dataset", split="val", image_size=128)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        predict(model, val_loader, device, args)


if __name__ == "__main__":
    main()
