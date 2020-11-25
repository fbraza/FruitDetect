from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from keras.models import load_model
import cv2
import numpy as np


class ClassifPredictionModel:
    def __init__(self, path_model, path_classes, detection_window=None):
        """
        Instantiate an object which encapsulates all data and API necessary to
        run the predicitions

        Parameters:
        -----------
        - path_model: str, path to the model file generated
        after training
        - path_classes: str, path to the file containing the list
        of classes name
        - detection_window : list with top-left x, top-left y and
        bottom-right x, bottom-right y coordonates of the image
        window used for detection

        Attributes:
        -----------
        - classes: list[str], list containing names of the classes
        - model: Keras model used for detection
        - detection_rows : slice of rows of the detection windows
        - detection_columns : slice of columns of the detection windows
        - detection_upleft : upleft coordinates of the detection windows
        - detection_downright : downright coordinates of the detection windows
        """
        self.classes = self.class_names(path_classes)
        self.model = load_model(path_model)
        self.detection_rows = slice(
            detection_window[1],
            detection_window[3]
        )
        self.detection_columns = slice(
            detection_window[0],
            detection_window[2]
        )
        self.detection_upleft = (detection_window[0], detection_window[1])
        self.detection_downright = (detection_window[2], detection_window[3])

    def class_names(self, path):
        """
        Method defined to generate a list of class names' strings

        Parameters:
        -----------
        - path: str, path to the obj.names file

        Return:
        -------
        - None
        """
        with open(path, "rt") as file:
            return file.read().rstrip("\n").split("\n")

    def prepare_image(self, image):
        """
        Setter method defined to prepare image for classification and store it.

        Parameters:
        -----------
        - image: image object (width image, height image, channels image)

        Return:
        -------
        - preprocessed image
        """
        # crop the image with detection windows
        img_crop = image[self.detection_rows, self.detection_columns]
        # resize to expected size for model
        img_resized = cv2.resize(img_crop, (224, 224))
        img_expanded = np.expand_dims(img_resized, axis=0)
        preprocessed_image = preprocess_input_vgg(img_expanded)
        return preprocessed_image

    def predict_and_identify(self, image, threshold=0.5):
        """
        Method defined to get and output predictions on video frames.

        Parameters:
        -----------
        - image: numpy array representation of the image
        - threshold: float, 0.5 by default
        Return:
        -------
        - pred_class_idx : list of classes
        - pred_proba_value : probability of the prediction
        - text : prediction value as a string
        - image is modified adding the windows_detection as a red rectangle
          on the image
        """

        preprocessed_image = self.prepare_image(image)

        # the prediction is done here and put into pred variable
        pred = self.model.predict(preprocessed_image)

        # add a red rectangle (detection windows to the image)
        cv2.rectangle(image, self.detection_upleft,
                      self.detection_downright, (0, 0, 255), 0)

        # look at the higher probability among up/down/unknown
        # if prediction under threshold% probability, return -1/No Prediction
        pred_class_idx = -1
        if np.max(pred, axis=1)[0] > threshold:
            pred_class_idx = np.argmax(pred, axis=1)[0]
            pred_proba_value = pred[0][pred_class_idx]
            text = "{}".format(self.classes[pred_class_idx])

        else:
            pred_class_idx = -1
            text = "No Prediction"
            pred_proba_value = -1
            # store the prediction
        return pred_class_idx, pred_proba_value, text
