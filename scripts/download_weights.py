import os
import zipfile

import gdown


def download():
    print("Downloading AVSS checkpoint...")
    gdown.download(id="1WC5J6mWtWBtV0BG1l6JeKK4bYUekb2wA")
    os.makedirs("../data/other", exist_ok=True)
    os.rename("R12.pth", "../data/other/R12.pth")

    print("Downloading SS checkpoint...")
    gdown.download(id="13ZO4Duixv7xP9xqnBzT6aZMLjxH2YB1W")
    os.rename("no_video_model.pth", "../data/other/no_video_model.pth")

    print("Downloading lipreading model...")
    gdown.download(id="1YLWlxIu5xctnUds1yq4PMRi6MGhaXR2C")
    os.rename("lipreading_model.pth", "../data/other/lipreading_model.pth")


if __name__ == "__main__":
    download()
