'''
Below is a simple SAHI inference. It takes a path to the image as well as the YOLO model, and generates predictions.

There is a sample image in this directory to run inferencing on.

Batch inferences are also possible. (See testbed for an example)

Example invocation:
python run_sahi_inference.py --model_path "../Trained_models/yolov8l_trained_on_tiled_640_set.pt" --image_path "./sample_visdrone_image.jpg"
'''
import argparse
import sahi.predict

parser = argparse.ArgumentParser(description='This script runs a simple SAHI inference')
parser.add_argument('--model_path', type=str, required=False, default="../Trained_models/yolov8l_trained_on_tiled_640_set.pt", help='Path to the .pt YOLO model')
parser.add_argument('--image_path', type=str, required=False, default="./sample_visdrone_image.jpg", help='Path to the image to make predictions on')

# SAHI configurations (optional)
parser.add_argument('--slice_height_and_width', type=int, required=False, default=640, help='Slice size to use. (default 640)')
parser.add_argument('--overlap_height_and_width_ratio', type=float, required=False, default=0.2, help='Overlap ratio for slices. (default 0.2)')

args = parser.parse_args()

confidence_threshold = 0.5

# run
result = sahi.predict.predict(
    # constant
    model_type = 'yolov8',
    model_device = 'cuda:0',
    export_pickle = False,
    verbose = 0,

    # configs
    model_path = args.model_path,
    source = args.image_path,
    project = '.',
    name = 'output/output',

    model_confidence_threshold = confidence_threshold,

    # use SAHI slicing
    no_sliced_prediction=False,

    # SAHI parameters
    slice_height = args.slice_height_and_width,
    slice_width = args.slice_height_and_width,
    overlap_height_ratio = args.overlap_height_and_width_ratio,
    overlap_width_ratio = args.overlap_height_and_width_ratio,

    # postprocessing parameters
    postprocess_type="NMS",
    postprocess_match_metric="IOU",
    postprocess_match_threshold=confidence_threshold,
    force_postprocess_type=True
)
