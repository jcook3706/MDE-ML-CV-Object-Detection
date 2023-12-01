# SAHI Testbed

This script tests a yolo model with an option of using SAHI or not.
It makes inferences on a specified directory of images and calcuates mAP based on comparing the model's predictions to the ground-truth labels.

# How to run
## Setup:
- Clone this repo
- Find the location of your VisDrone dataset, along with the VisDrone.yaml that describes it.
- Follow the next step to make the key bug fix edit to the SAHI library.

### SAHI bug fix
- There is a bug in the SAHI library in which the confidence threshold configuration is not passed to the YOLO model. This is important for mAP calculation, as a small threshold is needed.
- Locate the location of your SAHI library, specifically the modes/yolov8.py file.
- Note: if using ARC and the same configuration, this will be at `~/py38venv/lib/python3.8/site-packages/sahi/models/yolov8.py`
- On line 61, replace the next lines of code with this

        # edited this to change confidence threshold
        prediction_result = self.model(image[:, :, ::-1], verbose=False, conf=self.confidence_threshold, max_det=1000)  # YOLOv8 expects numpy arrays to have BGR
        prediction_result = [
            result.boxes.data[result.boxes.data[:, 4] >= self.confidence_threshold] for result in prediction_result
        ]
- Save the file. You are now ready to run

## Run:
Run with `python run_testbed.py` with parameters described below.
Required arguments:
- `--model_path`: Path to your yolo model (.pt file)
- `--dataset_path`: Path to your dataset description file. (.yaml file)
- `--run_nickname`: The name of the folder to write results to.

Optional arguments:
- `--use_sahi`: If present, SAHI is run on top of the yolo model.
- `--conf`: A float that represents the minimum confidence threshold for the model. Use 0.001 when benchmarking for mAP, and 0.5 or antoher number when looking for visual results on what a production model would detect. (default 0.5)
- `--dataset_partition`: Partition of dataset to use: val or test. Use val (a samller set) for quicker runs, and test for offical testing.
- `--skip_ground_truth`: If present, skips converting ground truth labels to correct format (can be used after first run)
- `--postprocess_type`: The postprocessing type to use. (default GREEDYNMM. Can also be NMM or NMS)
- `--postprocess_match_metric`: The postprocessing match metric to use. (default IOS. Can also be IOU)
- `--postprocess_match_threshold`: The postprocessing match threshold to use. (default 0.5)
- `--slice_height_and_width`: Slice size to use. (default 256)
- `--overlap_height_and_width_ratio`: Overlap ratio for slices. (default 0.2)
- `--model_size`: Size of the yolo model being used. Has no effect except it is saved to output. 

### Example usages
`python run_testbed.py --model_path ./yolov8l_trained_9_15_23.pt --dataset_path /home/mdkattwinkel/mde/VisDrone.yaml --use_sahi --conf 0.01 --run_nickname testrun --skip_ground_truth`

`python run_testbed.py --model_path ./yolov8l_trained_9_15_23.pt --model_size=large --dataset_path /home/mdkattwinkel/mde/VisDrone.yaml --conf 0.001 --use_sahi --run_nickname myBestConfig-lrg-s700-NMS-iou --skip_ground_truth --postprocess_type NMS --postprocess_match_metric IOU --slice_height_and_width 700`

## Output:
Results will be saved to `testbed_sahi/runs/{run_nickname}` including:
- `output.json`: All stats about the run including mAP and all configurations.
- `map.log`: The mAP calculation results.
- `exp/visuals`: Visual results of the predictions on each image.

## References:
The calculations folder is taken and adapted from https://github.com/Cartucho/mAP