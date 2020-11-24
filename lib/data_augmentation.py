import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm


def get_images_and_box_files_names(path):
    """
    Function defined to load the names of the image and yolo bounding
    boxes files. It loads the images and yolo names separately.

    Args:
    -----
    - path: string of the folder's path

    Returns:
    --------
    - Tuple[List[str], List[str]]
    """
    files_names = os.listdir(path)
    images = sorted([names for names in files_names if names.endswith(".jpg")])
    bboxes = sorted([names for names in files_names if names.endswith(".txt")])
    return (images, bboxes)


def get_labels_and_coordinates(path):
    """
    Function defined to load yolo bounding boxes txt files
    as a numpy ndarray from which we extract the class label
    and the coordinates. These are stored in disinct lists and
    both returned in a tuples.

    Args:
    -----
    - path: str, file name

    Returns:
    --------
    - Tuple[List[int], List['ndarray']
    """
    data, labels, coordinates = np.loadtxt(path), [], []
    if len(data.shape) == 1:
        data = data.reshape(1, -1)

    for row in data:
        labels.append(row[0])
        coordinates.append(row[1:])
    return (labels, coordinates)


def set_new_files_names(original_file_name, tag, *extensions):
    """
    Helper function that process an original name takes out its extension and
    new files with new extension.The newly generated name can then be incorpo-
    rate in a new path to save files.

    Args:
    -----
    - original_file_name: str, file name
    - tag: int, a number to tag the file name
    - extensions: *args, strings for extension files

    Returns:
    --------
    - List[str]
    """
    temp = original_file_name.replace(f".{extensions[0]}", "")
    return [f"{temp}_{tag:02}.{extension}" for extension in extensions]


def get_data_from_pipeline(pipeline, image, coordinates, labels):
    """
    Function defined to apply the data augmentation pipeline and retrive image,
    bounding boxes coordinates and corresponding class labels.

    Args:
    -----
    - pipeline: albumentations.Compose Object
    - image: ndarray, image to be augmented
    - coordinates: List['ndarray[float]'], the coordinates of the bounding box
    - labels: List[float], labels of each class

    Returns:
    --------
    - Tuple['ndarray', List['ndarray'], List[float]]
    """
    aug_data = pipeline(image=image, bboxes=coordinates, class_labels=labels)
    return aug_data['image'], aug_data['bboxes'], aug_data['class_labels']


def save_image_bbox_data(path_to_save_data, image, image_name, coordinates,
                         labels, yolo_name):
    """
    Function defined
        - to save the augmented / tranformed image on the disk in a directory
        - Process the labels and transformed coordinates into an numpy array
        - Save the data encapsulated in the array in a txt file respecting the
          yolo format

    Args:
    -----
    - path_to_save_data: str, folder where to save augmented data
    - image: ndarray, image to be saved
    - image_name: str, image file name
    - coordinates: List['ndarray[float]'], returned bboxes from the pipeline
    - labels: List[float], labels of each class
    - yolo_name: str, yolo txt file name

    Returns:
    --------
    - None
    """
    cv2.imwrite(f"{path_to_save_data}/{image_name}", image)
    new_coordinates = np.array(coordinates)
    new_yolo_bbox = np.insert(new_coordinates, 0, labels, 1)
    np.savetxt(f"{path_to_save_data}/{yolo_name}",
               new_yolo_bbox, ["%i", "%f", "%f", "%f", "%f"])


def augment_and_save(path_to_get_data, path_to_save_data,
                     number_of_tranformation=10):
    """
    Function defined to apply an image / rounding boxes transformation pipeline
    and save the corresponding files.

    Args:
    -----
    - path_to_get_data: str, the folder path where untouched data is
    - path_to_save_data: str, the folder path where to save augmented data
    - number_of_tranformation: int, number of transformation to perform

    Returns:
    --------
    -  None
    """
    images_names, yolo_names = get_images_and_box_files_names(path_to_get_data)
    augmentation_pipeline = a.Compose(
            [A.Resize(256, 256), A.Flip(0.35), A.Blur(7, False, 0.35)],
            A.BboxParams('yolo', ['class_labels'])
            )
    # Iterate through each image
    for idx, name in enumerate(images_names):
        image_path = path_to_get_data + '/' + name
        image = cv2.imread(image_path)
        yolo_file_path = path_to_get_data + '/' + yolo_names[idx]
        labels, coordinates = get_labels_and_coordinates(yolo_file_path)
        # Generate x tranformation of the images
        for i in tqdm(range(number_of_tranformation)):
            new_image_name, new_yolos_name = set_new_files_names(
                                                name, i, "jpg", "txt"
                                                )
            # Catch error due to unproper labelling
            try:
                new_image, new_coordinates, labels = get_data_from_pipeline(
                                                        augmentation_pipeline,
                                                        image, coordinates,
                                                        labels
                                                        )
            except ValueError as e:
                print("**** Error Message ****\n")
                print(f"{e}\n")
                print(f"""Invalid transformation of box:
                          {str(new_coordinates)}\n""")
                print(f"Image: {new_image_name}\n")
                continue
            # Same each image to jpg with its corresponding coordinates
            save_image_bbox_data(path_to_save_data, new_image, new_image_name,
                                 new_coordinates, labels, new_yolos_name)
