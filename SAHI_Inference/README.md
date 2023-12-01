### SAHI Inference
This folder contains the tools to make a simple SAHI inference. For more detailed explanations of SAHI setups and configurations, look to the linked documents and the testbed folder.

`run_sahi_inference.py` is the main script for this folder. It takes a path to an image, as well as a path to a YOLO model. It generates predicions from this and saves them as a png file.

`sample_visdrone_image.jpg` is a sample image included in this directory for running this script quickly.

It currently uses default configurations but that can be changed by adding the `--slice_height_and_width` and `--overlap_height_and_width_ratio` arguments to the command line.

Example invocation:
`python run_sahi_inference.py --model_path "../Trained_models/yolov8l_trained_on_tiled_640_set.pt" --image_path "./sample_visdrone_image.jpg"`