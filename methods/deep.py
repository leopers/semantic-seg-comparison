import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets.oxford_pets import OxfordPetsSegmentation
from models.unet import UNet

# U-net pipeline


def train_cnn(
    dataset: OxfordPetsSegmentation,
    batch_size: int = 32,
    num_epochs: int = 50,
    plot_loss: bool = True,
):
    """Trains the implemented model on the Oxford Pets dataset."""

    # Selecting GPU acceleration if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epoch_losses = []

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        )

        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item() * images.size(0)
            running_loss += batch_loss

            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataset)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "../models/trained/unet_oxfordpets.pth")

    if plot_loss:
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss per Epoch")
        plt.grid(True)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run Unet on OxfordPets")
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--train", type=bool, default=True)
    # parser.add_argument("--output-dir", type=str, default="outputs/classical")
    parser.add_argument("--image-size", type=int, default=128)
    # parser.add_argument("--pad", type=int, default=10)
    # parser.add_argument("--index", type=int, default=0)
    # parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.train:
        train_dataset = OxfordPetsSegmentation(
            args.dataset_root, split="train", image_size=args.image_size
        )
        train_cnn(train_dataset)

    # ds = OxfordPetsSegmentation(
    #     root=args.dataset_root, split="val", image_size=args.image_size
    # )
    #
    # if args.all:
    #     for i in range(len(ds)):
    #         process_sample(i, ds, args.output_dir, args.pad)
    # else:
    #     process_sample(args.index, ds, args.output_dir, args.pad)


if __name__ == "__main__":
    main()
