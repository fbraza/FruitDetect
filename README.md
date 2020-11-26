# Fruit detection using deep learning and human-machine interaction



## Introduction

Most of the retails markets have self-service systems where the client can put the fruit but need to navigate through the system's interface to select and validate the fruits they want to buy. The good delivery of this process highly depends on human interactions and actually holds some trade-offs: heavy interface, difficulty to find the fruit we are looking for on the machine, human errors or intentional wrong labeling of the fruit and so on.

Herein the purpose of our work is to propose an alternative approach to identify fruits in retail markets. More specifically we think that the improvement should consist of a faster process leveraging an user-friendly interface. One aspect of this project is to delegate the fruit identification step to the computer using deep learning technology. The interaction with the system will be then limited to a validation step performed by the client. This step also relies on the use of deep learning and gestural detection instead of direct physical interaction with the machine. Theoretically this proposal could both simplify and speed up the process to identify fruits and limit errors by removing the human factor.

We will report here the fundamentals needed to build such detection system. Regarding hardware, the fundamentals are two cameras and a computer to run the system . A fruit detection model has been trained and evaluated using the fourth version of the You Only Look Once (YOLOv4) object detection architecture. The human validation step has been established using a convolutional neural network (CNN) for classification of thumb-up and thumb-down. The model has been written using Keras, a high-level framework for Tensor Flow.  For both deep learning systems the predictions are ran on an backend server while a front-end user interface will output the detection results and presents the user interface to let the client validate the predictions.



## Methodology & Results

#### Dataset
In this project we aim at the identification of 4 different fruits: tomatoes, bananas, apples and mangoes. From these we defined 4 different classes by fruits: *single fruit*, *group of fruit*, *fruit in bag*, *group of fruit in bag*. An additional class for an empty camera field has been added which puts the total number of classes to 17. A data set of 20 to 30 images per class has been generated using the same camera as for predictions. In total we got 338 images. Example images for each class are provided in Figure 1 below. 

![Figure 1](https://github.com/fbraza/DSTI_Python_Labs/blob/readme/assets/Figure_1.png)

*Figure 1: Representative pictures of our fruits without and with bags*

The Computer Vision and Annotation Tool ([CVAT](https://github.com/openvinotoolkit/cvat)) has been used to label the images and export the bounding boxes data in YOLO format.

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
Each image went through 150 distinct rounds of transformations which brings the total number of images to 50700. Our images have been spitted into training and validation sets at a 9|1 ratio. The full code can be seen [here](https://github.com/fbraza/DSTI_Python_Labs/blob/master/lib_data_augmentation/data_augmentation.py) for data augmentation and [here](https://github.com/fbraza/DSTI_Python_Labs/blob/master/lib_data_augmentation/split_train_test.py) for the creation of training & validation sets.

#### Fruit detection model training with YOLOv4
For fruit detection we used the YOLOv4 architecture whom backbone network is based on the CSPDarknet53 ResNet. YOLO is a one-stage detector meaning that predictions for object localization and classification are done at the same time. Additionally and through its previous iterations the model significantly improves by adding Batch-norm, higher resolution, anchor boxes, objectness score to bounding box prediction and a detection in three granular step to improve the detection of smaller objects. From the user perspective YOLO proved to be very easy to use and setup. Indeed because of the time restriction when using the Google Colab free tier we decided to install locally all necessary drivers (NVIDIA, CUDA) and compile locally the [Darknet architecture](https://github.com/AlexeyAB/darknet). This has been done on a Linux computer running Ubuntu 20.04, with 32GB of RAM, NVIDIA GeForce GTX1060 graphic card with 6GB memory and an Intel i7 processor.

To evaluate the model we relied on two metrics: the **mean average precision** (mAP) and the **intersection over union** (IoU). The **average precision** (AP) is a way to get a fair idea of the model performance. It consists of computing the maximum precision we can get at different threshold of recall. Then we calculate the mean of these maximum precision. Now as we have more classes we need to get the AP for each class and then compute the mean again. This why this metric is named **mean average precision**. Object detection brings an additional complexity: what if the model detects the correct class but at the wrong location meaning that the bounding box is completely off. Surely this prediction should not be counted as positive. That is where the IoU comes handy and allows to determines whether the bounding box is located at the right location. Usually a threshold of 0.5 is set and results above are considered as good prediction. As such the corresponding mAP is noted **mAP@0.5**. The principle of the IoU is depicted in Figure 2.

![](https://github.com/fbraza/DSTI_Python_Labs/blob/readme/assets/Figure_2.jpg)

*Figure 2: Intersection over union principle*

The final results that we present here stems from an iterative process that prompted us to adapt several aspects of our model notably regarding the generation of our dataset and the splitting into different classes. We did not modify the architecture of YOLOv4 and run the model locally using some custom configuration file and pre-trained [weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137) for the convolutional layers (`yolov4.conv.137`). Once everything is set up we just ran:

```bash
./darknet detector train data/obj.data cfg/yolov4-custom-dsti.cfg yolov4.conv.137
```

We ran five different experiments and present below the result from the last one. The training last 4 days to reach a loss function of 1.1 (Figure 3A). To assess our model on validation set we used the `map` function from the darknet library with the final weights generated by our training:

```bash
./darknet detector map data/obj.data cfg/yolov4-custom-dsti.cfg backup/yolov4-custom-dsti_final.weights
```

The results yielded by the validation set were fairly good as mAP@50 was about 98.72% with an average IoU of 90.47% (Figure 3B). We always tested our results by recording on camera the detection of our fruit to get a real feeling of the accuracy of our model as illustrated in Figure 3C.

![Figure 3](https://github.com/fbraza/DSTI_Python_Labs/blob/readme/assets/Figure_3.png)

*Figure 3: Loss function (A). Metrics on validation set (B). Representative detection of our fruits (C)*

Below you can see a couple of short videos that illustrates how well our model works for fruit detection.

- **All fruits**

<img src="https://github.com/fbraza/DSTI_Python_Labs/blob/readme/assets/Video_1.gif" alt="Video_1" style="zoom:50%;" />

- **Apples inside bag**

<img src="https://github.com/fbraza/DSTI_Python_Labs/blob/readme/assets/Video_2.gif" alt="Video_2" style="zoom:50%;" />

- **Bananas inside bag**

<img src="https://github.com/fbraza/DSTI_Python_Labs/blob/readme/assets/Video_3.gif" alt="Video_3" style="zoom:50%;" />

#### Thumb detection model training with Keras

Pictures of thumb up (690 pictures), thumb down (791 pictures) and empty background pictures (347) on different positions and of different sizes have been taken with a webcam and used to train our model. Affine image transformations have been used for data augmentation (rotation, width shift, height shift). We use transfer learning with a **vgg16 neural network** imported with `imagenet` weights but without the top layers. We then add `flatten`, `dropout`, `dense`, `dropout` and `predictions` layers. The activation function of the last layer is a sigmoid function. The model has been ran in jupyter notebook on Google Colab with GPU using the free-tier account. The final architecture of our CNN neural network is described in the table below.

| **Layer (type)**            | **Output Shape**        | **Param \#** |
| --------------------------- | ----------------------- | ------------ |
| input\_1 (InputLayer)       | \[(None, 224, 224, 3)\] | 0            |
| block1\_conv1 (Conv2D)      | (None, 224, 224, 64)    | 1792         |
| block1\_conv2 (Conv2D)      | (None, 224, 224, 64)    | 36928        |
| block1\_pool (MaxPooling2D) | (None, 112, 112, 64)    | 0            |
| block2\_conv1 (Conv2D)      | (None, 112, 112, 128)   | 73856        |
| block2\_conv2 (Conv2D)      | (None, 112, 112, 128)   | 147584       |
| block2\_pool (MaxPooling2D) | (None, 56, 56, 128)     | 0            |
| block3\_conv1 (Conv2D)      | (None, 56, 56, 256)     | 295168       |
| block3\_conv2 (Conv2D)      | (None, 56, 56, 256)     | 590080       |
| block3\_conv3 (Conv2D)      | (None, 56, 56, 256)     | 590080       |
| block3\_pool (MaxPooling2D) | (None, 28, 28, 256)     | 0            |
| block4\_conv1 (Conv2D)      | (None, 28, 28, 512)     | 1180160      |
| block4\_conv2 (Conv2D)      | (None, 28, 28, 512)     | 2359808      |
| block4\_conv3 (Conv2D)      | (None, 28, 28, 512)     | 2359808      |
| block4\_pool (MaxPooling2D) | (None, 14, 14, 512)     | 0            |
| block5\_conv1 (Conv2D)      | (None, 14, 14, 512)     | 2359808      |
| block5\_conv2 (Conv2D)      | (None, 14, 14, 512)     | 2359808      |
| block5\_conv3 (Conv2D)      | (None, 14, 14, 512)     | 2359808      |
| block5\_pool (MaxPooling2D) | (None, 7, 7, 512)       | 0            |
| flatten (Flatten)           | (None, 25088)           | 0            |
| dropout (Dropout)           | (None, 25088)           | 0            |
| dense (Dense)               | (None, 128)             | 3211392      |
| dropout\_1 (Dropout)        | (None, 128)             | 0            |
| predictions (Dense)         | (None, 3)               | 387          |

Monitoring **loss function** and **accuracy** (precision) on both training and validation sets has been performed to assess the efficacy of our model. We can see that the training was quite fast to obtain a robust model. As soon as the fifth *Epoch* we have a abrupt decrease of the value of the loss function for both training and validation sets which coincides with an abrupt increase of the accuracy (Figure 4).  

![Figure 4](https://github.com/fbraza/DSTI_Python_Labs/blob/readme/assets/Figure_4.png)

*Figure 4: Accuracy and loss function for CNN thumb classification model with Keras*

It took around 30 *Epochs* for the training set to obtain a stable loss very closed to 0 and an very high accuracy closed to 1. Our test with camera demonstrated that our model was robust and working well.

- **Thumb down detection**

<img src="https://github.com/fbraza/DSTI_Python_Labs/blob/readme/assets/Video_4.gif" alt="Video_4" style="zoom:50%;" />

- **Thumb up detection**

<img src="https://github.com/fbraza/DSTI_Python_Labs/blob/readme/assets/Video_5.gif" alt="Video_5" style="zoom:50%;" />

#### Server-side and client side application architecture

The architecture and design of the app has been thought with the objective to appear autonomous and simple to use. it is supposed to lead  the user in the right direction with minimal interaction calls (Figure 4). The user needs to put the fruit under the camera, reads the proposition from the machine and validates or not the prediction by raising his thumb up or down respectively. 

![workflow](https://github.com/fbraza/DSTI_Python_Labs/blob/readme/assets/Figure_5.png)

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
- a frontend client with HTML files send by the Flask server and JavaScript modules that treat messages coming from the backend.

In our situation the interaction between backend and frontend is bi-directional. First the backend reacts to client side interaction (e.g., press a button, fill a form). Second we also need to modify the behavior of the frontend depending on what is happening on the backend. In this regard we complemented the Flask server with the [Flask-socketio](<https://www.shanelynn.ie/asynchronous-updates-to-a-webpage-with-flask-and-socket-io/>) library to be able to send such messages from the server to the client. This is well illustrated in two cases: 

- The approach used to handle the image streams generated by the camera where the backend deals directly with image frames and send them subsequently to the client side. This approach circumvents any web browser compatibility issues as `png` images are sent to the browser. Treatment of the image stream has been done using the Open-CV library and the whole logic has been encapsulated into a python class `Camera`. The full code can be read [here](https://github.com/fbraza/DSTI_Python_Labs/blob/master/lib_prediction/camera.py).

- The approach used to treat fruits and thumb detection and send the results to the client where models and predictions are respectively loaded and analyzed on the backend then results are directly send as messages to the frontend. Based on the message the client needs to display different pages. An example of the code can be read below for result of the thumb detection. The full code can be read [here](https://github.com/fbraza/DSTI_Python_Labs/blob/master/static/js/application.js).

  ```javascript
    // receive details from server
      socket.on('newnumber', function(msg) {
          console.log("Received number" + msg.number);
          // maintain a list of ten numbers
          if (msg.number == 0)
          {
              result\_string = '\<p\> Thumb UP selected \</p\>';
              setTimeout(function(){
                  window.location.href = "/ticket\_printing";
              }, 4000);
  ```

The final product and its usage can be seen below (**video / gif**).



## Discussion and perspectives

The final product we obtained revealed to be quite robust and easy to use. We managed to develop and put in production locally two deep learning models in order to smoothen the process of buying fruit in a super-market with the objectives mentioned in our introduction. However as every proof-of-concept our product still lacks some critical aspects and needs to be improved. 

Regarding the detection of fruits the final result we obtained stems from a iterative process through which we experimented a lot. A major point of confusion for us was the establishment of a proper dataset. In our first attempt we generated a bigger dataset with 400 photos by fruit. These photos were taken by each member of the project using different smart-phones. Interestingly while we got a bigger dataset after data augmentation the model's predictions were pretty unstable in reality despite yielding very good metrics at the validation step. That is why we decided to start from scratch and generated a new dataset using the camera that will be used by the final product (our webcam). Unexpectedly doing so and with less data lead to a more robust model of fruit detection with still nevertheless some unresolved edge cases. Indeed in all our photos we limited the maximum number of fruits to 4 which makes the model unstable when more similar fruits are on the camera. To illustrate this we had for example the case where above 4 tomatoes the system starts to predict apples! Additionally we  need more photos with fruits in bag to allow the system to generalize better. Altogether this strongly indicates building a bigger dataset with photos shot in the real context could resolve some of these points. 

While we do manage to deploy locally an application we still need to consolidate and consider some aspects before putting this project to production. This raised many questions and discussions in the frame of this project and fall under the umbrella of several topics that include deployment, continuous development of the data set, tracking, monitoring & maintenance of the models.

For the deployment part we should consider testing our models using less resource consuming neural network architectures. For fruit we used the full YOLOv4 as we were pretty comfortable with the computer power we had access to. However we should anticipate that devices that will run in market retails will not be as resourceful. As a consequence it will be interesting to test our application using some lite versions of the YOLOv4 architecture and assess whether we can get similar prediction and user experience. Similarly we should also test the usage of the Keras model on litter computers and see if we yield similar results.  [Raspberry Pi](https://www.raspberrypi.org/) devices could be interesting machine to imagine a final product for the market. They are cheap and have been shown to be handy devices to deploy lite models of deep learning.

Once the model is deployed one might think about how to improve it and how to handle edge cases raised by the client. Imagine the following situation. One client put the fruit in front of the camera and put his thumb down because the prediction is wrong. Firstly we definitively need to implement a way out in our application to let the client select by himself the fruits especially if the machine keeps giving wrong predictions. Secondly what can we do with these wrong predictions ? We could actually save them for later use. We could even make the client indirectly participate to the labeling in case of wrong prediction. Indeed when a prediction is wrong we could implement the following feature: save the picture, its wrong label into a database (probably a No-SQL document database here with timestamps as a key), and the real label that the client will enter as his way-out. Later the engineers could extract all the wrong predicted images, relabel them correctly and re-train the model by including the new images. This immediately raises another questions: when should we train new model ? Some monitoring of our system should be implemented. One might think to keep track of all the prediction made by the device and daily or weekly monitor some easy metrics: `number of right total predictions / number of total predictions`, `number of wrong total predictions / number of total predictions`. These metrics can then be declined by fruits. Establishing such strategy would imply the implementation of some data warehouse with the possibility to quickly generate reports that will help to take decisions regarding the update of the model.

To conclude here we are confident in achieving a reliable product with high potential. Then, convincing supermarkets to adopt the system should not be too difficult as the cost is limited when the benefits could be very significant. Moreover, an example of using this kind of system exists in the catering sector with **Compass company** since 2019. It is applied to dishes recognition on a tray. Thousands of different products can be detected, and the bill is automatically output. The waiting time for paying has been divided by 3. More broadly, automatic object detection rather than manual interaction is certainly a future success technology. Of course, the autonomous car is the current most impressive project. But a lot of simpler applications in the everyday life could be imagined. The cost of cameras has become dramatically low, the possibility to deploy neural network architectures on small devices, allowing considering this tool like a new powerful human machine interface. A further idea would be to improve the thumb recognition process by allowing all fingers’ detection, making possible to count. A more advanced approach to communicate with the system could be also envisioned. Voice control has already been used for a few years...Giving ears and eyes to machines definitely makes them closer to human behavior.