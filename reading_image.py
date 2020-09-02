import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import albumentations as A


def read_image(path: str, opencv_color_flag: int) -> np.ndarray:
    """
    Function defined to read an image and output it.

    Args:
        path: string of the image path
        opencv_color_flag:  choose -1 (load image as such) , 0 (loads image in grayscale) or 1 (loads image in BGR)

    Returns:
        np.ndarray
    """
    my_image = cv2.imread(filename = path, flags = opencv_color_flag)
    return my_image


def show_image(img: np.ndarray) -> None:
    """
    Function defined to output an image using matplotlib

    Args:
        img: np.array representing a previously loaded image

    Returns:
        None
    """
    plt.figure(figsize = (15, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # By default cv2 output image in BGR channels.
    plt.show()

if __name__ == '__main__':
    test     = read_image(path = "Data/20200827_084342.jpg", opencv_color_flag = -1)
    pipeline = A.Compose([
    A.Resize(height = 256, width = 256),
    A.HorizontalFlip(p = 0.5),
    A.VerticalFlip(p = 0.5)
    ])

    resized = pipeline(image = test)
    show_image(resized['image'])
