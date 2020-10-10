# Project structure

```bash
# root directory
.
├── code
│   ├── __init__.py
│   ├── io_process.py
│   ├── lib
│   ├── README.md
│   └── requirements.txt
└── data
    ├── model_training
    ├── original_input
    ├── transformed_input
    └── yolo_data
    
# code directory
.
├── __init__.py
├── io_process.py
├── lib
│   ├── data_augmentation.py
│   ├── __init__.py
│   ├── prediction.py
│   └── split_train_test.py
├── README.md
└── requirements.txt

# data directory
.
├── model_training
│   ├── Fruit_detection.ipynb
│   ├── yolov4-custom.cfg
│   └── yolov4-obj_best.weights
├── original_input
│   ├── All
│   ├── Apple
│   ├── Banana
│   ├── Mango
│   └── Tomato
├── transformed_input
└── yolo_data
    ├── obj # folder with training data
    ├── obj.data
    ├── obj.names # file with class names
    └── test # folder with test data
```

