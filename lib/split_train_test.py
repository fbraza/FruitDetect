import os
import shutil
from random import shuffle


class DataSplitor:
    def __init__(self, source_data, train_dest, test_dest):
        self.source_data = source_data
        self.train_dest = train_dest
        self.test_dest = test_dest
        self.train = None
        self.test = None

    def set_train_test(self, split_factor=0.8):
        """
        Setter method defined to split the unique file names into two lists:
        one that will be used for training set and the other for test set.
        The list is shuffled and splitted using the split factor.

        Target:
        -------
        - Instance of DataSplitor

        Args:
        -----
        - split_factor: float, between 0 and 1, by default 0.8

        Returns:
        --------
        - tuple[list[string]]
        """
        names = self.__unique_file_names()
        shuffle(names)
        self.train = names[:int(len(names) * split_factor)]
        self.test = names[int(len(names) * split_factor):]

    def __unique_file_names(self):
        """
        'Private' method defined to select unique names from our transformed
        images. After data augmentation two types of files are generated: jpg
        and txt with the same name but respective extension. Here the idea is
        to extract the common root. We use a set data structure to have the
        unique names.

        Target:
        -------
        - Instance of DataSplitor

        Returns:
        --------
        - list[str]
        """
        files_names = sorted(os.listdir(self.source_data))
        unique_root = set()
        for name in files_names:
            unique_root.add(name[:-4])
        return list(unique_root)


def generate_yolo_inputs(source_data, split_factor=0.8):
    """
    This function automatizes the creation of the yolo files train.txt and
    test.txt that contain the names of the image files that will be process
    by the model. This function leverages the DataSplitor API to write files
    and copy images and yolo coordinates into new folders ready to be used by
    the Yolo model.

    Args:
    -------
    - source_data: str, path where the augmented img and yolo coordinates files
    are located
    - split_factor: float, between 0 and 1, by default 0.8

    Returns:
    --------
    - None
    """
    spliter = DataSplitor(source_data, "yolo_data/obj", "yolo_data/test")
    spliter.set_train_test(split_factor)
    train = open("yolo_data/train.txt", "w+")
    test = open("yolo_data/test.txt", "w+")
    # Iterate through the file names for train
    for idx, name in enumerate(spliter.train):
        write_train_txt(train, name, len(spliter.train), idx)
        copy_data(name, source_data, spliter.train_dest)
    # Iterate through the file names for test
    for idx, name in enumerate(spliter.test):
        write_test_txt(test, name, len(spliter.test), idx)
        copy_data(name, source_data, spliter.test_dest)


def write_train_txt(file_to_write, file_name, size_file_list, index):
    """
    This helper function defined to instruct the I/O process necessary to write
    the yolo files and copy images in the right folder. It will process each
    file name on the train list encapsulated as an attribute of an DataSplitor

    Args:
    -------
    - file_to_rite: File, file to write the name of the iamges to be processed
    by the model
    - file_name: str, file name that will be written
    - size_file_list: int, size of the file list
    - index: index of the file in the list

    Returns:
    --------
    - None
    """
    img_name = "{}.jpg".format(file_name)
    if index == size_file_list - 1:
        file_to_write.write("data/obj/{}".format(img_name))
    else:
        file_to_write.write("data/obj/{}\n".format(img_name))


def write_test_txt(file_to_write, file_name, size_file_list, index):
    """
    This helper function defined to instruct the I/O process necessary to write
    the yolo files and copy images in the right folder. It will process each
    file name on the test list encapsulated as an attribute of an DataSplitor

    Args:
    -------
    - file_to_rite: File, file to write the name of the iamges to be processed
    by the model
    - file_name: str, file name that will be written
    - size_file_list: int, size of the file list
    - index: index of the file in the list

    Returns:
    --------
    - None
    """
    img_name = "{}.jpg".format(file_name)
    if index == size_file_list - 1:
        file_to_write.write("data/test/{}".format(img_name))
    else:
        file_to_write.write("data/test/{}\n".format(img_name))


def copy_data(file_name, source, destination):
    """
    This helper function defined to instruct the I/O process necessary to copy
    the yolo files and copy images in the right folder. It will process each
    file name on the test list encapsulated as an attribute of an DataSplitor

    Args:
    -------
    - file_name: str, file name that will be written
    - source: str, path pointing to the source folder
    - destination: str, path pointing to the destination foler

    Returns:
    --------
    - None
    """
    img_to_move = "{}.jpg".format(file_name)
    txt_to_move = "{}.txt".format(file_name)
    shutil.copy("{}/{}".format(source, img_to_move), destination)
    shutil.copy("{}/{}".format(source, txt_to_move), destination)
