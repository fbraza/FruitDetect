# Fruit detection using deep learning and human-machine interaction

## Introduction

Most of the retails markets have self-service systems where the client can put the fruit but need to navigate through the
system's interface to select and validate the fruits they want to buy. The good delivery of this process highly depends on
human interactions and actually holds some trade-offs: heavy interface, difficulty to find the fruit we are looking for on
the machine, human errors or intentional wrong labeling of the fruit and so on.

Herein the purpose of our work is to propose an alternative approach to identify fruits in retail markets. More
specifically we think that the improvment should consist of a faster process leveraging an user-friendly interface. One
aspect of this project is to delegate the fruit identification step to the computer using deep learning technology. The
interaction with the system will be then limited to a validation step perfomed by the client. This step also relies on
the use of deep learning and gestual detection instead of direct physical interaction with the machine. Theoritically this
proposal could both simplify and speed up the process to identify fruits and limit errors by removing the human factor.

We will report here the fundamentals needed to build such detection system. Regarding hardware, the fundamentals are two
cameras and a computer to run the system **(Fig1: Draw the system)**. A fruit detection model has been trained and
evaluated using the fourth version of the You Only Look Once (YOLOv4) object detection architecture. The human validation
step has been establish using **[ERIC PLEASE COMPLETE HERE]**. For both deep learning systems the predictions are ran on an
backend server while a front-end user interface will output the detection results and presents the user interface to let the
client validate the predictions.

## Methodology

*Dataset* - In this project we aim at the identification of 4 different fruits: tomatoes, bananas, apples and mangos.
From these we defined 4 different classes by fruits: single fruit, group of fruit, fruit in bag, group of fruit in bag.
An additional class for an empty camera field has been added which puts the total number of classes to 17. A data set of
**[number]** images per class has been generated using the same camera as for pediction. Example images for each class 
are provided in Figure 2. The Computer Vision and Annotation Tool (CVAT) has been used to label the images and export
the bounding boxes data in YOLO format.

*Data augmentation* - We used traditional tranformations that combined affine image transformations and color modification.
These tranformations have been performed using the [Albumentations](https://github.com/albumentations-team/albumentations) python library. This library leverages `numpy`, `opencv` and `imgaug` python lirabries through an easy to use API.
```python
albumentation_pipeline_object.Compose([A.Resize(256, 256), A.Flip(0.35), A.Blur(7, False, 0.35)],
                                       A.BboxParams('yolo', ['class_labels']))
```
For each image 
*Fruit detection with YOLOv4* -
*Thumb detection with TensorFlow* -

## Experiment and results

## Discussion and perspectives
