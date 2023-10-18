'''
This script tests a yolo model with an option of using SAHI or not.
It makes inferences on a specified directory of images and calcuates mAP based on comparing predictions to ground-truth.
See README for instructions on how to run

example usage
python run_testbed.py --model_path /home/mdkattwinkel/mde/yolov8l_trained_9_15_23.pt --dataset_path /home/mdkattwinkel/mde/VisDrone.yaml --use_sahi --conf 0.01 --run_nickname testrun --skip_ground_truth
'''
import argparse

# Get arguments from the command line
parser = argparse.ArgumentParser(description='This script tests a yolo model with an option of using SAHI or not')
parser.add_argument('--model_path', type=str, required=True, help='Path to the .pt YOLO model')
parser.add_argument('--model_size', type=str, required=True, help='Size of the yolo model being used. Saved to output')
parser.add_argument('--run_nickname', type=str, required=True, help='User defined name - will be used for log file name')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to .yaml file that describes the dataset')
parser.add_argument('--skip_ground_truth', action='store_true', required=False, help='Skip converting ground truth if its already done')
parser.add_argument('--use_sahi', action='store_true', required=False, help='Use SAHI')
parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
args = parser.parse_args()

##########################################################################################
# Configurations: CHANGE THESE TO AFFECT SAHI RUN
##########################################################################################
configurations = {
    "sahi": args.use_sahi,
    "conf_threshold": args.conf,
    
    # Currently hardcoded configurations. Change these to affect model
    "sahi_configurations": {
        "slice_height" : 256, 
        "slice_width" : 256,
        "overlap_height_ratio": 0.2,
        "overlap_width_ratio" : 0.2,
        "postprocess_class_agnostic" : True

        # TODO: can add more sahi configurations that we find out affect the results
    }
}

print("Setting up test...")
import os
import yaml
import shutil
import subprocess
import sahi.predict
import glob
import pickle
import re
import json

##########################################################################################
# Usage Verification
##########################################################################################

# verify arguments
if not os.path.exists(args.model_path):
    print(f"Model path [{args.model_path}] does not exist.")
    exit()
if not os.path.exists(args.dataset_path):
    print(f"Dataset path [{args.dataset_path}] does not exist.")
    exit()

# Check if the logfile exists
logs_folder = "./runs"
os.makedirs(logs_folder, exist_ok=True)
run_dir = os.path.join(logs_folder, args.run_nickname)
if os.path.exists(run_dir):
    print(f"The run directory '{run_dir}' already exists. Please choose a different name.")
    exit()

# get info from YAML and verify
with open(args.dataset_path, 'r') as yaml_file:
    dataset_yaml_data = yaml.safe_load(yaml_file)
val_path = dataset_yaml_data['val']
names = dataset_yaml_data['names']

dataset_images_dir_path = os.path.join(val_path, 'images')
dataset_labels_dir_path = os.path.join(val_path, 'labels')
if not os.path.exists(dataset_images_dir_path):
    print(f"Couldn't find Validtion path [{os.path.join(val_path, 'images')}] found in given dataset file [{args.dataset_path}].")
    exit()
if not os.path.exists(dataset_labels_dir_path):
    print(f"Couldn't find Validtion path [{os.path.join(val_path, 'labels')}] found in given dataset file [{args.dataset_path}].")
    exit()

calculation_ground_truth_path = "./calculations/input/ground-truth"
if not os.path.exists(calculation_ground_truth_path):
    print(f"[{calculation_ground_truth_path}] does not exist. You have the wrong file structure in this directory")
    exit()

##########################################################################################
# Ground truth conversion
##########################################################################################
if not args.skip_ground_truth:
    # move labels into calculations
    shutil.rmtree(calculation_ground_truth_path)
    shutil.copytree(dataset_labels_dir_path, calculation_ground_truth_path)

    # convert labels from the dataset into the correct format for the testbed to read
    print("Converting ground-truth labels to correct format...")
    subprocess.run(["python", "calculations/scripts/extra/convert_gt_yolo.py", "--images_path", dataset_images_dir_path], check=True)

##########################################################################################
# Model inferencing
##########################################################################################
# run the model's inferencing on each image from the dir specified in the dataset.yaml
# optionally use SAHI
print("Running Inferencing on images with the Model...")
# configs
output_dir = run_dir
use_sahi = configurations["sahi"] # optionally use SAHI
confidence_threshold= configurations["conf_threshold"] # confidence threshold
sahi_configs = configurations["sahi_configurations"]

# run
result = sahi.predict.predict(
    # constant
    model_type = 'yolov8',
    model_device = 'cuda:0',
    export_pickle = True,
    verbose = 0,

    # configs
    model_path = args.model_path,
    source = dataset_images_dir_path,
    project = output_dir,

    # parameters that may affect results - not exhaustive
    model_confidence_threshold = confidence_threshold,

    # use SAHI?
    no_sliced_prediction=not use_sahi,

    # SAHI parameters that may affect results - not exhaustive
    slice_height = sahi_configs["slice_height"], 
    slice_width = sahi_configs["slice_width"],
    overlap_height_ratio = sahi_configs["overlap_height_ratio"],
    overlap_width_ratio = sahi_configs["overlap_width_ratio"],
    postprocess_class_agnostic = sahi_configs["postprocess_class_agnostic"]
)
print("Finished making predictions!")

##########################################################################################
# Extracting results
##########################################################################################
print('Extracting results...')
# Unpickle inferences, write them in correct format to calculation dir
detections_dir = "./calculations/input/detection-results"
shutil.rmtree(detections_dir)
os.makedirs(detections_dir)
# Code adatpted from pickle.py written by Christopher
file_list = glob.glob(output_dir + '/exp/pickles/*.pickle')
for pickle_file in file_list:
    # Grab image name in order to use it to 
    image_name = os.path.basename(pickle_file)
    image_name = os.path.splitext(image_name)[0] + ".txt"
    # Load data (deserialize)
    with open(pickle_file, 'rb') as handle:
        unserialized_data = pickle.load(handle)
    write_file = open(os.path.join(detections_dir, image_name), "x")
    for item in unserialized_data:
        bbox = item.bbox
        left = bbox.minx
        right = bbox.maxx
        top = bbox.miny
        bottom = bbox.maxy
        category = item.category.name
        confidence = item.score.value
        predict_info = category + " " + str(confidence) + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + '\n'
        write_file.write(predict_info)
print('Finished extracting results!')

##########################################################################################
# Calculating mAP
##########################################################################################
print('Calculating mAP...')
# calculate mAP and inform user
output_log_path = f"{run_dir}/map.log"
command = f"python calculations/main.py 2>&1 | tee {output_log_path}"
subprocess.run(command, shell=True, check=True)

output_json_path = f"{run_dir}/output.json"
print(f'Finished Calculations. Saving output summary to {output_json_path}')

##########################################################################################
# Saving summary json
##########################################################################################

# Regular expression pattern to match lines in the log file
pattern = r"([\d.]+)% = (.+) AP"

class_specific_ap = {}
mAP = None

# Read and parse log file for AP and mAP
with open(output_log_path, 'r') as log_file:
    for line in log_file:
        match = re.match(pattern, line)
        if match:
            percentage, label = match.groups()
            class_specific_ap[label.strip()] = float(percentage)
        elif line.startswith("mAP"):
            mAP = float(re.search(r"([\d.]+)%", line).group(1))

# Create the final dictionary with mAP and class-specific AP
output_data = {
    "run_name": args.run_nickname,
    "model_path": args.model_path,
    "model_size": args.model_size,
    "configurations" : configurations,
    "mAP": mAP,
    "class_specific_AP": class_specific_ap
}

# Save the results as JSON
with open(output_json_path, 'w') as output_file:
    json.dump(output_data, output_file, indent=4)

print(f"Summary has been saved to {output_json_path}.")
print('Done.')

# TODO: clean up