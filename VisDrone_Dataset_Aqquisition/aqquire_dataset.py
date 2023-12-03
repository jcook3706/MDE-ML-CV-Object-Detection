import requests
from tqdm import tqdm
import zipfile
import io
import shutil
import os
import gdown

## Get the Labels in YOLO format

print("Downloading YOLO labels...")
url = "https://github.com/adityatandon/VisDrone2YOLO/archive/refs/heads/main.zip"
zip_file_path = "VisDroneYOLOLabels.zip"
extracted_folder_path = "VisDroneYOLOLabels"

response = requests.get(url, stream=True)

# Check if the request was unsuccessful
if response.status_code != 200:
    print(f"Failed to download. Status code: {response.status_code}")

# show progress
with open(zip_file_path, 'wb') as file, tqdm(
        desc=zip_file_path,
        total=int(response.headers.get('content-length', 0)),
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
) as bar:
    for data in response.iter_content(chunk_size=1024):
        bar.update(len(data))
        file.write(data)

print(f"Downloaded: {zip_file_path}")

# Unzip the file
print("Unzipping YOLO labels...")
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

print(f"Unzipped to: {extracted_folder_path}")

# Clean up the zip file
os.remove(zip_file_path)


# move the needed parts of the repo to the correct spot
print("Moving YOLO labels to correct spot...")
source_labels_path = "VisDroneYOLOLabels/VisDrone2YOLO-main"

def remove_and_copy_labels(source_folder, target_folder):
    # Remove the target labels folder if it exists
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
        print(f"Removed existing folder: {target_folder}")

    # Copy the source labels folder to the target location
    shutil.move(os.path.join(source_labels_path, source_folder), target_folder)
    print(f"Copied labels from {source_labels_path}/{source_folder} to {target_folder}")

remove_and_copy_labels("VisDrone2019-DET-train/labels", "VisDrone/VisDrone2019-DET-train/labels")
remove_and_copy_labels("VisDrone2019-DET-val/labels", "VisDrone/VisDrone2019-DET-val/labels")
remove_and_copy_labels("VisDrone2019-DET-test-dev/labels", "VisDrone/VisDrone2019-DET-test-dev/labels")

shutil.rmtree("VisDroneYOLOLabels")


# Download all the visdrone images
print("Downloading YOLO images...")

# download val
file_id = "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59"
zip_file_name = "VisDrone2019-DET-val.zip"
gdown.download(id=file_id, output=zip_file_name, quiet=False)
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall("VisDrone2019-DET-val")
os.remove(zip_file_name)
target_folder = "VisDrone/VisDrone2019-DET-val/images"
source_folder = "VisDrone2019-DET-val/VisDrone2019-DET-val/images"
# Remove the target images folder if it exists
if os.path.exists(target_folder):
    shutil.rmtree(target_folder)
    print(f"Removed existing folder: {target_folder}")
# move the source images folder to the target location
shutil.move(source_folder, target_folder)
shutil.rmtree("VisDrone2019-DET-val")
print(f"Moved images from {source_folder} to {target_folder}")

# download test-dev
file_id = "1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V"
zip_file_name = "VisDrone2019-DET-test-dev.zip"
gdown.download(id=file_id, output=zip_file_name, quiet=False)
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall("VisDrone2019-DET-test-dev")
os.remove(zip_file_name)
target_folder = "VisDrone/VisDrone2019-DET-test-dev/images"
# NOTE: this could be "VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev/images" sometimes for some reason
source_folder = "VisDrone2019-DET-test-dev/images"
# Remove the target images folder if it exists
if os.path.exists(target_folder):
    shutil.rmtree(target_folder)
    print(f"Removed existing folder: {target_folder}")
# move the source images folder to the target location
shutil.move(source_folder, target_folder)
shutil.rmtree("VisDrone2019-DET-test-dev")
print(f"Moved images from {source_folder} to {target_folder}")

# download train (largest)
file_id = "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn"
zip_file_name = "VisDrone2019-DET-train.zip"
gdown.download(id=file_id, output=zip_file_name, quiet=False)
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall("VisDrone2019-DET-train")
os.remove(zip_file_name)
target_folder = "VisDrone/VisDrone2019-DET-train/images"
source_folder = "VisDrone2019-DET-train/VisDrone2019-DET-train/images"
# Remove the target images folder if it exists
if os.path.exists(target_folder):
    shutil.rmtree(target_folder)
    print(f"Removed existing folder: {target_folder}")
# move the source images folder to the target location
shutil.move(source_folder, target_folder)
shutil.rmtree("VisDrone2019-DET-train")
print(f"Moved images from {source_folder} to {target_folder}")

print("Successfully downlaoded labels and images for VisDrone datasets.")