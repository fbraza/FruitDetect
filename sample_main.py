import albumentations as A
import cv2
import numpy as np
import LabelFileTreatment as lbl
#import reading_image

# Load Image and transform it from BGR to RGB
#image = read_image("/samples/TwoTomatoesImage.jpg",1)
image = cv2.imread("samples/TwoTomatoesImage.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Print image to check
#show_image(image)

# setup a simple data augmentation : horizontal flip
transform = A.Compose([
    A.HorizontalFlip(p=1),
], bbox_params=A.BboxParams(format='yolo', min_area=1024, min_visibility=0.1, label_fields=['class_labels']))

# create the object for labeltreatment
labelFile = lbl.LabelTreatment('samples/TwoTomatoesImage.txt','samples/TwoTomatoesImage_flip.txt','yolo')
# get list expected for Albumentations from the label file
bboxes,class_labels = labelFile.getBoxObjectFromFile()

# check boxes coordinates and labels list before transformation
print(bboxes,class_labels)

# use Albumentation to transform :
# * the image
# * the box coordinates
# * the label list
transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
transformed_image = transformed['image']
transformed_bboxes = transformed['bboxes']
transformed_class_labels = transformed['class_labels']

# check boxes coordinates and labels list before transformation
print(transformed_bboxes,transformed_class_labels)

# check image after transformation
#show_image(transformed_image)

# Save new image to disk
transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("samples/TwoTomatoesImage_flip.jpg",transformed_image)
# Save transformed labels to disk
labelFile.putBoxObjectToFile(transformed_bboxes,transformed_class_labels)

