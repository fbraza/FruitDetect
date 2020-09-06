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