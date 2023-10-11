# SAHI Testbed

This script tests a yolo model with an option of using SAHI or not.
It makes inferences on a specified directory of images and calcuates mAP based on comparing the model's predictions to the ground-truth labels.

# How to run
## Setup:
- Clone this repo
- Find the location of your VisDrone dataset, along with the VisDrone.yaml that describes it.
- Follow the next step to make the key bug fix edit to the SAHI library.

### SAHI bug fix
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
- `--use_sahi`: If present, SAHI is run on top of the yolo model. As of now, SAHI configurations are handled in the code.
- `--skip_ground_truth`: If present, skips converting ground truth labels to correct format (can be used after first run)
- `--conf`: A float that represents the minimum confidence threshold for the model (default 0.5)

### Example usage
`python run_testbed.py --model_path /home/mdkattwinkel/mde/yolov8l_trained_9_15_23.pt --dataset_path /home/mdkattwinkel/mde/VisDrone.yaml --use_sahi --conf 0.01 --run_nickname testrun --skip_ground_truth`

## Output:
Results will be saved to `testbed_sahi/runs/{run_nickname}` including:
- `map.log`: The mAP calculation results.
- `exp/visuals`: Visual results of the predictions on each image.

## References:
The calculations folder is taken and adapted from https://github.com/Cartucho/mAP