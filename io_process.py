if __name__ == "__main__":
    from lib.data_augmentation import augment_and_save
    from lib.split_train_test import generate_yolo_inputs
    print("Data Processing starts...\n")
    augment_and_save("data/original_input/All", "data/transformed_input", 30)
    print("Preparing data for Darknet - Yolo\n")
    generate_yolo_inputs("data/transformed_input", 0.75)
    print("Data augmention, File I/O processing for Yolo are done\n")
    print("You may now use the notebook to train your model")
