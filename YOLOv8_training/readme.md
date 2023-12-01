# How to train the YOLOv8 from scratch
This directory contains the information to train a YOLOv8 model on the VISDRONE dataset

## Setup
The main bulk of setup has to do with setting up the dataset
1. Aqquire the dataset (including train and val) and put it in a directory of your choice
  - The dataset needs to be in format to be read by yolo. (See https://github.com/adityatandon/VisDrone2YOLO/tree/main and `visDrone2YOLO.py` for the annotations and labels. The actual images will need to be downloaded from another source)
2. Edit `VisDrone.yaml` to put the appropriate paths to train and val dataset at the top of the file

## Run
1. Adjust hyperparameters (image size, batch size, YOLOv8 variations, etc.) if necessary in `train.py`.
2. Run `train.py` on your desired machine.
  - We recommend that the training is done on a machine that has high-computing resources, as the training will require a great amount of computation resources and time.

## Results
1. After the training completes, YOLOv8 will create a folder `runs/detect/train`. To access the trained model, take a look at `weights/best.pt` for the best checkpoint file of the trained model.
  - Note that `yolov8_example_train.out` contains the example training output you can expect to see once the model finishes training.