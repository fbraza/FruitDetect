if __name__ == "__main__":
    from data_augmentation import augment_and_save
    from split_train_test import generate_yolo_inputs
    in_ = "../../data/original_input/All_new"
    out_ = "../../data/transformed_input"
    print("Data Processing starts...\n")
    augment_and_save(in_, out_, 150)
    print("Preparing data for Darknet - Yolo\n")
    generate_yolo_inputs(out_, 0.95)
    print("Data augmention, File I/O processing for Yolo are done\n")
    print("You may now use the notebook to train your model")
