# Fruit detection using deep learning and human-machine interaction



## Introduction

Most of the retails markets have self-service systems where the client can put the fruit but need to navigate through the system's interface to select and validate the fruits they want to buy. The good delivery of this process highly depends on human interactions and actually holds some trade-offs: heavy interface, difficulty to find the fruit we are looking for on the machine, human errors or intentional wrong labeling of the fruit and so on.

Herein the purpose of our work is to propose an alternative approach to identify fruits in retail markets. More specifically we think that the improvement should consist of a faster process leveraging an user-friendly interface. One aspect of this project is to delegate the fruit identification step to the computer using deep learning technology. The interaction with the system will be then limited to a validation step performed by the client. This step also relies on the use of deep learning and gestural detection instead of direct physical interaction with the machine. Theoretically this proposal could both simplify and speed up the process to identify fruits and limit errors by removing the human factor.

We will report here the fundamentals needed to build such detection system. Regarding hardware, the fundamentals are two cameras and a computer to run the system **(Fig1: Draw the system)**. A fruit detection model has been trained and evaluated using the fourth version of the You Only Look Once (YOLOv4) object detection architecture. The human validation step has been establish using **[ERIC PLEASE COMPLETE HERE]**. For both deep learning systems the predictions are ran on an backend server while a front-end user interface will output the detection results and presents the user interface to let the client validate the predictions.



## Methodology

#### Dataset
In this project we aim at the identification of 4 different fruits: tomatoes, bananas, apples and mangoes. From these we defined 4 different classes by fruits: *single fruit*, *group of fruit*, *fruit in bag*, *group of fruit in bag*. An additional class for an empty camera field has been added which puts the total number of classes to 17. A data set of 20 to 30 images per class has been generated using the same camera as for predictions. In total we got 338 images. Example images for each class are provided in Figure 2. The Computer Vision and Annotation Tool ([CVAT](https://github.com/openvinotoolkit/cvat)) has been used to label the images and export the bounding boxes data in YOLO format.

```Bash
# Example of labeling data in YOLO format

Class Index     x          y          width      height
---------------------------------------------------------
13   			0.438271   0.523156   0.179000   0.191354
13   			0.581010   0.358792   0.183646   0.205250
13   			0.688271   0.516125   0.203708   0.193042
9    			0.568677   0.433490   0.449063   0.356479

# x and y are the coordinates of the center point of the bounding box
# Values are normalized regarding size of the image
```



#### Data augmentation
We used traditional transformations that combined affine image transformations and color modifications. These transformations have been performed using the [Albumentations](https://github.com/albumentations-team/albumentations) python library. This library leverages `numpy`, `opencv` and `imgaug` python libraries through an easy to use API. The sequence of transformations can be seen below in the code snippet.
```python
A.Compose(
    [A.Resize(416, 416),
     A.Equalize(by_channels=True),
     A.RGBShiftr_shift_limit=(-30, 30), g_shift_limit=(-30, 30), b_shift_limit=(-30, 30), p=0.25),
     A.HorizontalFlip(p=0.35),
     A.VerticalFlip(p=0.35),
     A.ShiftScaleRotate(border_mode=cv2.BORDER_REPLICATE, p=0.35),
     A.RandomSnow(brightness_coeff=2.0, p=0.2)],
     A.BboxParams('yolo', ['class_labels'])
    )
```
Each image went through 150 distinct rounds of transformations which brings the total number of images to 50700. The full code for data augmentation can be seen [here]() **PUT THE LINK**

#### Fruit detection with YOLOv4
For fruit detection we used the YOLOv4 architecture whom backbone network is based on the CSPDarknet53 ResNet. YOLO is a one-stage detector meaning that predictions for object localization and classification are done at the same time. Additionally and through its previous iterations the model significantly improves by adding Batch-norm, higher resolution, anchor boxes, objectness score to bounding box prediction and a detection in three granular step to improve the detection of smaller objects. From the user perspective YOLO proved to be very easy to use and setup. Indeed because of the time restriction when using the Google Colab free tier we decided to install locally all necessary drivers (NVIDIA, CUDA) and compile locally the [Darknet architecture](https://github.com/AlexeyAB/darknet). This has been done on a Linux computer running Ubuntu 20.04, with 32GB of RAM, NVIDIA GeForce GTX1060 graphic card with 6GB memory and an Intel i7 processor.

To evaluate the model we relied on two metrics: the **mean average precision** (mAP) and the **intersection over union** (IoU). The **average precision** (AP) is a way to get a fair idea of the model performance. It consists of computing the maximum precision we can get at different threshold of recall. Then we calculate the mean of these maximum precision. Now as we have more classes we need to get the AP for each class and then compute the mean again. This why this metric is named **mean average precision**. Object detection brings an additional complexity: what if the model detects the correct class but at the wrong location meaning that the bounding box is completely off. Surely this prediction should not be counted as positive. That is where the IoU comes handy and allows to determines whether the bounding box is located at the right location. Usually a threshold of 0.5 is set and results above are considered as good prediction. As such the corresponding mAP is noted **mAP@0.5**. The principle of the IoU is depicted in Figure 3.

#### Thumb detection with Tensor Flow

#### Server-side and client side application architecture

The architecture and design of the app has been thought with the objective to appear autonomous and simple to use. it is supposed to lead  the user in the right direction with minimal interaction calls (Figure 4). The user needs to put the fruit under the camera, read the proposition from the machine and validate or not the prediction by raising his thumb up or down respectively. 

![workflow](https://github.com/fbraza/DSTI_Python_Labs/blob/master/assets/app_workflow.png)

*Figure 4: Application workflow*

For the predictions we envisioned 3 different scenarios:

1. The easiest one where nothing is detected. 
2. The scenario where one and only one fruit is detected.
3. The scenario where several fruit are detected by the machine

From these 3 scenarios we can have different possible outcomes:

1. Nothing is detected because no fruit is there or the machine cannot predict anything (very unlikely in our case)
2. One fruit is detected then we move to the next step where user needs to validate or not the prediction. If the user negates the prediction the whole process starts from beginning.
3. Several fruit are detected. The process restarts from the beginning and the user needs to put a uniform group of fruits. 

From a technical point of view the choice we have made to implement the application are the following:

- a backend server that runs locally using Flask.
- a frontend client with static HTML files updated by the Flask server and JavaScript modules that treat messages coming from the backend.

In our situation the interaction between backend and frontend is bi-directional. First the backend reacts to client side interaction (e.g., press a button, fill a form). Second we also need to modify the behavior of the frontend depending on what is happening on the backend. In this regard we complemented the Flask server with the [Flask-socketio](<https://www.shanelynn.ie/asynchronous-updates-to-a-webpage-with-flask-and-socket-io/>) library to be able to send such messages from the server to the client. Finally the image stream generated by the camera is handled on the backend then frames are sent to the client. This approach excludes any problem regarding web browser's compatibility as `png` image streams are sent to the client. Treatment of the image stream has been done using the Open-CV library and the whole logic has been encapsulated into a python class `Camera`. The full code can be read [here]() **LINK**.


## Experiment and results

## Discussion and perspectives
