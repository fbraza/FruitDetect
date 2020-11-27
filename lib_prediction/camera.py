from threading import Thread
from .predict_thumb import ClassifPredictionModel
from .predict_fruit import YoloPredictionModel, generate_blob
import numpy as np
import cv2


PATH_MODEL_THUMB = "config/tensorflow/weights_thumb_v1.h5"
PATH_CLASSES_THUMB = "config/tensorflow/thumb.classes"
PATH_CONFIG = "config/yolo/yolov4-custom-dsti.cfg"
PATH_WEIGHTS = "config/yolo/yolov4-custom-dsti_final.weights"
PATH_CLASSES = "config/yolo/obj.names"

classif_thumb = ClassifPredictionModel(PATH_MODEL_THUMB,
                                       PATH_CLASSES_THUMB,
                                       (190, 50, 440, 300))
yolo = YoloPredictionModel(PATH_CONFIG,
                           PATH_WEIGHTS,
                           PATH_CLASSES).set_backend_and_device()

# initialiaze a global variable for thread
thread = None


class Camera:
    def __init__(self, socketio=None, video_source=0, nb_predicted_images=60):
        """
        This function initialize the class Camera

        Args:
        -----
        - socketio : the socket object to communicate with the front end
        - video_source : the id of the camera. This should enable connection
        with several cameras at the same time to the server
        - nb_predicted_images : the number of prediction done before sending
        the result of the prediction

        Returns:
        --------
        -  None
        """
        self.video_source = video_source
        self.socketio = socketio
        self.nb_predicted_images = nb_predicted_images
        self.camera = cv2.VideoCapture(self.video_source)
        self.frames = []
        self._predictions = []
        self.isrunning = False

    def __del__(self):
        """
        This function delete the class Camera

        Returns:
        --------
        - None
        """
        self.camera.release(self.camera.video_source)

    def run(self):
        """
        This function start a thread in which the frame
        acquisition will be done

        Returns:
        --------
        - None
        """
        global thread
        if thread is None:
            self.frames = []
            self._predictions = []
            self.camera.open(self.video_source)
            thread = Thread(target=self._capture_loop, daemon=True)
            self.isrunning = True
            thread.start()

    def _capture_loop(self):
        """
        This function updates the attribute self.frames with the
        image read from the camera flow
        Args:
        -----
        - None
        Returns:
        --------
        - None
        """
        while self.isrunning:
            v, im = self.camera.read()
            if v:
                self.frames = im
        self.camera.release()

    def stop(self):
        """
        This function stops the thread

        Returns:
        --------
        - None
        """
        self.isrunning = False

    def get_frame_thumb(self):
        """
        This function realizes the prediction and show the result
        of this prediction on the frame

        Returns:
        --------
        - None
        """
        global thread
        if len(self.frames) > 0:
            frame = self.frames
            # get the prediction from class ClassifPredictionModel
            pred_class_idx, pred_proba_value, text = (
                classif_thumb
                .predict_and_identify(
                    frame, 0.5)
            )
            frame = cv2.flip(frame, 1)
            # store the prediction on a list that will be evaluated 
            # at the end of the prediction process
            self._predictions.append(pred_class_idx)
            # add the result of the prediction to the image
            # displayed on the webpage
            cv2.putText(frame,
                        text,
                        (35, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        5)
            img = cv2.imencode('.png', frame)[1].tobytes()
            # if number of analysed images is reached, send result to frontend
            if len(self._predictions) > self.nb_predicted_images:
                self.stop()
                thread, self.frames = None, []
                # count the number of up thumb and down thumb
                up_count = np.sum(np.array(self._predictions) == 0)
                down_count = np.sum(np.array(self._predictions) == 1)
                # check that prediction has been done on more then 10% of images
                # if yes, send the prediction 0 for thumb up, 1 for thumb down
                # if no, send the code 2 for no prediction
                total = (self.nb_predicted_images / 10)
                if (up_count >= down_count) and (up_count > total):
                    self.socketio.emit('newnumber',
                                       {'number': 0},
                                       namespace='/start')
                elif (up_count < down_count) and (down_count > total):
                    self.socketio.emit('newnumber',
                                       {'number': 1},
                                       namespace='/start')
                else:
                    self.socketio.emit('newnumber',
                                       {'number': 2},
                                       namespace='/start')
        else:
            with open("./assets/0.jpg", "rb") as f:
                img = f.read()
        return img

    def get_frame_fruit(self):
        """
        This function realizes the prediction and shows the results
        of this prediction on the frame.

        Returns:
        --------
        - None
        """
        global model_thumb
        global thread
        if len(self.frames) > 0:
            img2 = self.frames
            img2 = cv2.flip(img2, 1)
            # tranform frames into blobs
            blob_input = generate_blob(img2)
            # blobs as yolo inputs
            yolo.ingest_input(blob_input)
            # Get output obects
            yolo.get_output_layers_names()
            output = yolo._forward()
            # Predictions
            classe, index, proba = yolo.predict_and_identify(img2, output, 0.5)
            result, classe_unique = result_fruit(classe, index, proba)
            # store the result on a list
            self._predictions.append(result)
            # encode the image in a .png file
            img = cv2.imencode('.png', img2)[1].tobytes()
            # if number of analysed images is reached, send result to frontend
            if len(self._predictions) > self.nb_predicted_images:
                # stop camera
                self.stop()
                thread = None
                self.frames = []
                # analyse the detection list and return one detection answer
                detection = detection_fruit(classe_unique, self._predictions)
                # add 10 to create an unique code for the frontend
                self.socketio.emit(
                    'newnumber', {'number': 10+detection}, namespace='/start')
        else:
            with open("./assets/0.jpg", "rb") as f:
                img = f.read()
        return img


def result_fruit(classe, index, proba):
    """
    This function analyse the prediction and return a consolidation
    of the predictions done

    Args:
    -----
    - classe : list of classes
    - index : list with the index of fruit detected
    - proba : probability of the detection (not used)

    Returns:
    --------
    - fruit detected : -1 -> no detection,
    0:3 -> index of the fruit, 4 -> several fruits detected
    - list of all unique indexes
    """
    classe_1st_word = [(txt.split())[0] for txt in classe]
    classe_unique = list(dict.fromkeys(classe_1st_word))
    list_nb_fruit = np.zeros(len(classe_unique))
    list_nb_indexes = [index.count(x) for x in range(len(classe))]
    for i, txt in enumerate(classe_1st_word):
        # do not take in account the 'Blank' detection
        if txt != "Blank":
            list_nb_fruit[classe_unique.index(txt)] += list_nb_indexes[i]
    # get the maximum number of detection for a fruit
    max_fruit = np.max(list_nb_fruit)
    # count the number of fruit with the maximum of detection
    if max_fruit != 0:
        res = np.sum(list_nb_fruit == max_fruit)
    else:
        res = 0
    # if only one fruit with maximum : this is the detection
    if res == 1:
        fruit = np.argmax(list_nb_fruit)
    # if more then one fruit with maximum then several fruits detected
    elif res > 1:
        fruit = len(classe_unique) - 1
    # else no detection
    else:
        fruit = -1
    # print(fruit)
    return fruit, classe_unique


def detection_fruit(classe, prediction):
    """
    This function analyses the list of prediction and return a
    consolidation of the predictions done

    Args:
    -----
    - classe : list of classes
    - prediction : list of predictions

    Returns:
    --------
    - fruit
    """
    nb_predicted_images = len(prediction)
    nb_fruit = len(classe)
    list_nb_indexes = [prediction.count(x) for x in range(-1, nb_fruit)]
    list_nb_fruit = list_nb_indexes[1:nb_fruit+1]
    fruit = list_nb_fruit.index(max(list_nb_fruit))
    nb_max = list_nb_fruit[fruit]
    if nb_max > (nb_predicted_images / 10):
        result = fruit
    else:
        result = -1
    return result
