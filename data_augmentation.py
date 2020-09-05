import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import albumentations as A
from tqdm import tqdm


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


def get_images_and_box_files_names(path: str) -> tuple:
    """
    Function defined to load the names of the image and yolo bounding boxes files
    It loads the images and yolo names separately.

    Args:
        path: string of the folder's path

    Returns:
        tuple
    """
    files_names = os.listdir(path) # this would raise an error if path does not exist
    images = sorted([names for names in files_names if names.endswith(".jpg")])
    bboxes = sorted([names for names in files_names if names.endswith(".txt")])
    return (images, bboxes)


def get_labels_and_coordinates(path: str) -> tuple:
    """
    Function defined to load yolo rounding boxes txt files
    as a numpy ndarray from which we extract the class label
    and the coordinates. These are stored in disinct lists and
    both returned in a tuples.

    Args:
        path: string of file name

    Returns:
        tuple
    """
    data = np.loadtxt(path)
    labels, coordinates = [], []
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    for row in data:
        labels.append(row[0])
        coordinates.append(row[1:])

    return (labels, coordinates)


def augment_and_save(path: str) -> None:
    """
    Function defined to apply an image / rounding boxes transformation pipeline
    and save the corresponding files.

    Args:
        path: string of file name

    Returns:
        tuple
    """
    # load files names and define a pre-established pipeline
    images_files_names, yolo_files_names = get_images_and_box_files_names(path)
    augmentation_pipeline = A.Compose(
        [A.Resize(height = 128, width = 128),
         A.Flip(p = 0.5),
         A.Normalize()],
         bbox_params = A.BboxParams(format = 'yolo', label_fields = ['class_labels'])
    )

    # iterate through names
    for idx, name in enumerate(images_files_names):
        # Read image
        image_path = path + '/' + name
        image = cv2.imread(image_path)
        # Extract labels and coordinates from yolo txt file
        yolo_file_path = path + '/' + yolo_files_names[idx]
        labels, coordinates = get_labels_and_coordinates(yolo_file_path)
        for i in tqdm(range(10)):
            name = name.replace(".jpg", "")
            new_image_name, new_yolos_name = f"{name}_{i:02}.jpg", f"{name}_{i:02}.txt"
            aug_data = augmentation_pipeline(image = image,
                                               bboxes = coordinates,
                                               class_labels = labels)
            new_image, new_coordinates, labels = aug_data['image'], aug_data['bboxes'], aug_data['class_labels']
            print(type(new_coordinates))
            break
            # write image on disk
            # cv2.imwrite("test_augmented_images/{new_image_name}", new_image)
            # # reform the array with labels and coordinates
            # new_bbox_yolo = np.insert()
            # np.savetxt("test_augmented_images/{new_yolos_name}")
        break


augment_and_save("test_40_photos/obj_train_data")