import os
import urllib.request
import tarfile
import shutil
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, block_num=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(block_num * block_size - self.n)


def download_with_progress(url, destination_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=os.path.basename(destination_path)
    ) as progress:
        urllib.request.urlretrieve(
            url, filename=destination_path, reporthook=progress.update_to
        )


def download_and_extract(url, destination_dir):
    filename = url.split("/")[-1]
    filepath = os.path.join(destination_dir, filename)

    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        download_with_progress(url, filepath)
    else:
        print(f"{filename} already exists. Skipping download.")

    print(f"Extracting {filename}...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=destination_dir)

    os.remove(filepath)
    print(f"Removed archive: {filename}")


def prepare_oxford_pets_dataset():
    os.makedirs("dataset", exist_ok=True)

    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = (
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    )

    download_and_extract(images_url, "dataset")
    download_and_extract(annotations_url, "dataset")

    # Move trimap segmentation masks to dataset/masks/
    source_masks_dir = "dataset/annotations/trimaps"
    target_masks_dir = "dataset/masks"

    if os.path.exists(target_masks_dir):
        shutil.rmtree(target_masks_dir)
    shutil.move(source_masks_dir, target_masks_dir)

    # Clean up unused annotation files
    annotations_dir = "dataset/annotations"
    if os.path.exists(annotations_dir):
        shutil.rmtree(annotations_dir)

    print("Oxford-IIIT Pet dataset is prepared.")
    print("Images directory: dataset/images/")
    print("Masks directory:  dataset/masks/")


if __name__ == "__main__":
    prepare_oxford_pets_dataset()
