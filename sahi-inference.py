## Below is a basic implementation of SAHI inference using a random image from the test dataset
## In order to calculate mAP values from SAHI we will need to add a lot of additional infromation

## batch inference are also possible using the "predict" function

import os
from ultralytics import YOLO

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

yolov8_model_path = "C:/Users/CV/Documents/College/Spring_2023/ECE_4805/Visdrone/yolov8m.pt"

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.4,
    device="cpu", # or 'cuda:0'
)

## result = get_prediction(read_image("Images/VisDrone2019-DET-test-dev/images/0000353_05500_d_0000199.jpg"), detection_model)

## result.export_visuals(export_dir="demo_data/")

## Image("/Images/VisDrone2019-DET-test-dev/images/0000353_05500_d_0000199.jpg")

result = get_sliced_prediction("Images/VisDrone2019-DET-test-dev/images/0000353_05500_d_0000199.jpg", 
                               detection_model,
                               slice_height = 256, 
                               slice_width = 256,
                               overlap_height_ratio= 0.2,
                               overlap_width_ratio = 0.2
                               )

result.export_visuals(export_dir="demo_data/")

object_prediction_list = result.object_prediction_list

print(object_prediction_list)