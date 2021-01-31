from sklearn.feature_extraction.image import _extract_patches
import numpy as np
import cv2
from datasets import download_and_prepare

def read_file(file, grayscale=False):
    """
    Loads and normalizes image.

    Parameters
        file (string): image file path
        grayscale (bool): True converts the image to grayscale

    Returns:
        (ndarray): image
    """
    image = cv2.imread(file)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.float32(image) / 255
    return image

files = download_and_prepare("utk-faces", "data")
Img = read_file(files[0])
print(Img.shape)
sample = _extract_patches(Img, patch_shape=(64,64,3), extraction_step=1)
print(sample.shape)