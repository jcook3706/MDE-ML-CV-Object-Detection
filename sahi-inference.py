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

yolov8_model_path = "C:/Users/CV/Documents/College/Spring_2023/ECE_4805/Visdrone/visdrone_L_pretrained.pt"

yolov8_config_file = "C:/Users/CV/Documents/College/Spring_2023/ECE_4805/Visdrone/VisDrone.yaml"
model_type='yolov8'
# model_path=yolov8_model_path,
confidence_threshold=0.5
device= "cuda:0" # "cpu" # or 'cuda:0'

## result = get_prediction(read_image("Images/VisDrone2019-DET-test-dev/images/0000353_05500_d_0000199.jpg"), detection_model)

## result.export_visuals(export_dir="demo_data/")

## Image("/Images/VisDrone2019-DET-test-dev/images/0000353_05500_d_0000199.jpg")

result = predict(
            model_type = model_type,
            model_path = yolov8_model_path,
            model_config_path = yolov8_config_file,
            model_device = device,
            model_confidence_threshold = confidence_threshold,
            source = "C:/Users/CV/Documents/College/Spring_2023/ECE_4805/Visdrone/Images/VisDrone2019-DET-test-dev/images", 
            slice_height = 256, 
            slice_width = 256,
            overlap_height_ratio= 0.2,
            overlap_width_ratio = 0.2,
            postprocess_class_agnostic = True,
            project = "C:/Users/CV/Documents/College/Spring_2023/ECE_4805/Visdrone/runs/predict",
            export_pickle = True,
            verbose = 0,
            no_sliced_prediction=True
        )

#print(object_prediction_list)