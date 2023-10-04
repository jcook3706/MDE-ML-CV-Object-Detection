import argparse
import os
import re
import json

parser = argparse.ArgumentParser(description='Test bed for machine learning models')
parser.add_argument('--model_path', type=str, required=True, help='Path to the .pt YOLO model') # hopefully include ONYXX
parser.add_argument('--model_nickname', type=str, required=True, help='User defined name')
parser.add_argument('--logfile_name', type=str, required=True, help='Name of the logfile')
parser.add_argument('--batch', type=int, required=False, help='Optional batch size')
args = parser.parse_args()

# Check if the logfile exists
logs_folder = "logs"
os.makedirs(logs_folder, exist_ok=True)
logfile_path = os.path.join(logs_folder, args.logfile_name + '.txt')
logjson_path = os.path.join(logs_folder, args.logfile_name + '.json')
if os.path.exists(logfile_path) or os.path.exists(logjson_path):
    print(f"The logfile '{logfile_path}' or '{logjson_path}' already exists. Please choose a different name.")
    exit()

# Build YOLO command with optional batch size
command = "yolo task=detect mode=val"
if args.batch:
    command += f" batch={args.batch}"
command += f" model={args.model_path} data=Visdrone.yaml 2>&1 | tee {logfile_path}"

# Run YOLO validation with the given model path and batch size
print("starting validation:")
os.system(command)

print()
print("Finished validation. Results written to {logfile_path} and {logjson_path}.")
print()

# convert to a json format
with open(logfile_path, 'r') as f:
    data = f.read()
# use regular expressions to extract relevant information
model_info = re.findall(r'Ultralytics\s(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', data)[0]
summary_info = re.findall(r'(\S+)\ssummary\s\(fused\):\s(\d+)\slayers,\s(\d+)\sparameters,\s(\d+)\sgradients,\s([\d\.]+)\sGFLOPs', data)[0]
class_info = re.findall(r'^\s+(\S+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', data, flags=re.MULTILINE)
speed_info = re.findall(r'Speed:\s([\d\.]+)ms\spreprocess,\s([\d\.]+)ms\sinference,\s([\d\.]+)ms\sloss,\s([\d\.]+)ms\spostprocess\w*\sper\simage', data)[0]
# create a dictionary to hold the extracted information
output = {
    'model_nickname': args.model_nickname,
    'model_name': model_info[0],
    'model_version': model_info[1],
    'python_version': model_info[2],
    'torch_version': model_info[3],
    'summary': {
        'num_layers': int(summary_info[1]),
        'num_parameters': int(summary_info[2]),
        'num_gradients': int(summary_info[3]),
        'GFLOPs': float(summary_info[4])
    },
    'classes': [],
    'speed': {
        'preprocess_time_ms': float(speed_info[0]),
        'inference_time_ms': float(speed_info[1]),
        'loss_time_ms': float(speed_info[2]),
        'postprocess_time_ms': float(speed_info[3])
    }
}
# add class information to the dictionary
for cls in class_info:
    output['classes'].append({
        'class_name': cls[0],
        'num_images': int(cls[1]),
        'num_instances': int(cls[2]),
        'P': float(cls[3]),
        'R': float(cls[4]),
        'mAP50': float(cls[5]),
        'mAP50-95': float(cls[6])
    })

# convert the dictionary to JSON. Print to stdout and file
print(json.dumps(output, indent=4))
with open(logjson_path, 'w') as f:
    json.dump(output, f, indent=4)