# MDE-ML-CV-Object-Detection

## Project Description
The goal of this project is to improve the small object detection performance, measured by mAP@0.5, on the VisDrone dataset using state-of-the art techniques.
This is accomplished using image tiling prior to model training in addition to Slicing Aided Hyper Inference (SAHI). The object detection model used is YOLOv8, the latest You Only Look Once model from Ultralytics as of December 2023.
SAHI and tiling do not alter the detection model itself, but rather are used atop the base model to improve performance for small objects.

We encountered challenges implementing this as a result of non-optimized slice sizing for the dataset and differences in mAP detection methods between YOLOv8 and SAHI. .
In the future, we would like to improve the mAP we achieved from this project using further techniques such as adaptive slice sizes that change based on estimated image depth and using relative object dimensions to eliminate false positives.

## Installation Instructions
1. Clone this repository
2. Run the provided VisDrone_Dataset_Acquisition script to download the VisDrone dataset
3. Install the Ultralytics and sahi libraries (we recommend using pip to do this)

## How to Use this Project
It is recommended that users perform model training on a powerful machine if possible. Some machines may not be capable of training the model due to limited memory.
We used university-owned compute clusters with significant resources for the training and it took over 24 hours to complete. 

This repository contains both trained models in addition to the code necessary to train new models. Trained models can be downloaded directly and used in the same way as the pre-trained YOLOv8 model provided by Ultralytics. 
Tutorials can be found online, such as the one at https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode.

To train a custom model, follow the directions provided in the YOLOv8_training folder. 

## Contributors 
This project was put together by a team of electrical and computer engineering students at Virginia Tech for their senior design project. 
Team members: Jimmy Cook, Michael Kattwinkel, Jalen Neal, Chris VanWinkle, Hayley Wisman, and Jay Yim
