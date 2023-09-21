# How to train the yolo untrained model
This directory contains the information to train a yolo model on the VISDRONE dataset

## Setup
The main bulk of setup has to do with setting up the dataset
1. Aqquire the dataset (including train and val) and put it in a directory of your choice
  - The dataset needs to be in format to be read by yolo. (See https://github.com/adityatandon/VisDrone2YOLO/tree/main for the annotations and labels. The actual images will need to be downloaded from another source)
2. Edit `VisDrone.yaml` to put the paths to train and val at the top of the file

## Run
1. You may want to test out the steps with an interactive job before submitting a batch job
2. Submit the batch job with `sbatch trainlarge.sh`

## Results
1. After the job completes, check the .out file for the path to the trained model, `best.pt`