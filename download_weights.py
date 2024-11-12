import os
import zipfile

import gdown


def download():
    # lipreading_model
    print("Downloading lipreading model...")
    gdown.download(id="1YLWlxIu5xctnUds1yq4PMRi6MGhaXR2C")
    os.makedirs("data/other", exist_ok=True)
    os.rename("lipreading_model.pth", "data/other/lipreading_model.pth")

    # dataset
    print("Downloading dataset...")
    gdown.download(id="1cFETrcGX3Q2y42TM8il8V-_KqA6P7f5P")
    os.makedirs("data/datasets", exist_ok=True)
    path_to_zip_file = "dla_dataset.zip"
    directory_to_extract_to = "data/datasets"
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    os.remove("dla_dataset.zip")


if __name__ == "__main__":
    download()
