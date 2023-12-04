import requests
from tqdm import tqdm
import zipfile
import shutil
import os
import gdown

######################################################################
# Download all zip files (if they don't exist)
######################################################################

# download val images
file_id = "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59"
zip_file_name_val = "zip_files/VisDrone2019-DET-val.zip"
if not os.path.exists(zip_file_name_val):
    print("Downloading val images...")
    gdown.download(id=file_id, output=zip_file_name_val, quiet=False)

# download test-dev images
file_id = "1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V"
zip_file_name_test = "zip_files/VisDrone2019-DET-test-dev.zip"
if not os.path.exists(zip_file_name_test):
    print("Downloading test-dev images...")
    gdown.download(id=file_id, output=zip_file_name_test, quiet=False)

# download train images
file_id = "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn"
zip_file_name_train = "zip_files/VisDrone2019-DET-train.zip"
if not os.path.exists(zip_file_name_train):
    print("Downloading train images...")
    gdown.download(id=file_id, output=zip_file_name_train, quiet=False)

# download all YOLO labels
zip_file_name_labels = "zip_files/VisDroneYOLOLabels.zip"
if not os.path.exists(zip_file_name_labels):
    print("Downloading all YOLO labels...")
    url = "https://github.com/adityatandon/VisDrone2YOLO/archive/refs/heads/main.zip"
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Failed to download. Status code: {response.status_code}")
        exit()
        # show progress
    with open(zip_file_name_labels, 'wb') as file, tqdm(
            desc=zip_file_name_labels,
            total=int(response.headers.get('content-length', 0)),
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            bar.update(len(data))
            file.write(data)


######################################################################
# Unzip zip files and move to correct location
######################################################################

# Unzip YOLO labels
print("Unzipping YOLO labels...")
extracted_folder_path_labels = "zip_files/VisDroneYOLOLabels"
with zipfile.ZipFile(zip_file_name_labels, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path_labels)
print(f"Unzipped to: {extracted_folder_path_labels}")

# move the needed parts of the repo to the correct spot
print("Moving YOLO labels to correct spot...")
source_labels_path = extracted_folder_path_labels + "/VisDrone2YOLO-main"

def remove_and_copy_labels(source_folder, target_folder):
    # Remove the target labels folder if it exists
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
        print(f"Removed existing folder: {target_folder}")

    # Copy the source labels folder to the target location
    shutil.move(os.path.join(source_labels_path, source_folder), target_folder)
    print(f"Moved labels from {source_labels_path}/{source_folder} to {target_folder}")

remove_and_copy_labels("VisDrone2019-DET-train/labels", "VisDrone/VisDrone2019-DET-train/labels")
remove_and_copy_labels("VisDrone2019-DET-val/labels", "VisDrone/VisDrone2019-DET-val/labels")
remove_and_copy_labels("VisDrone2019-DET-test-dev/labels", "VisDrone/VisDrone2019-DET-test-dev/labels")

# delete original folder 
shutil.rmtree(extracted_folder_path_labels)


# Unzip val images
print("Unzipping YOLO images (val)...")
extracted_folder_path_val = "zip_files/VisDrone2019-DET-val"
with zipfile.ZipFile(zip_file_name_val, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path_val)

# move images to correct spot
print("Moving val images to correct spot...")
target_folder = "VisDrone/VisDrone2019-DET-val/images"
source_folder = extracted_folder_path_val + "/VisDrone2019-DET-val/images"
# Remove the target images folder if it exists
if os.path.exists(target_folder):
    shutil.rmtree(target_folder)
    print(f"Removed existing folder: {target_folder}")
# move the source images folder to the target location
shutil.move(source_folder, target_folder)
shutil.rmtree(extracted_folder_path_val)
print(f"Moved images from {source_folder} to {target_folder}")

# Unzip test images
print("Unzipping YOLO images (test)...")
extracted_folder_path_test = "zip_files/VisDrone2019-DET-test-dev"
with zipfile.ZipFile(zip_file_name_test, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path_test)

# move images to correct spot
print("Moving test images to correct spot...")
target_folder = "VisDrone/VisDrone2019-DET-test-dev/images"
# NOTE: this could be "/VisDrone2019-DET-test-dev/images" sometimes for some reason
source_folder = extracted_folder_path_test + "/images"
# Remove the target images folder if it exists
if os.path.exists(target_folder):
    shutil.rmtree(target_folder)
    print(f"Removed existing folder: {target_folder}")
# move the source images folder to the target location
shutil.move(source_folder, target_folder)
shutil.rmtree(extracted_folder_path_test)
print(f"Moved images from {source_folder} to {target_folder}")


# Unzip train images
print("Unzipping YOLO images (train)...")
extracted_folder_path_train = "zip_files/VisDrone2019-DET-train"
with zipfile.ZipFile(zip_file_name_train, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path_train)

# move images to correct spot
print("Moving train images to correct spot...")
target_folder = "VisDrone/VisDrone2019-DET-train/images"
source_folder = extracted_folder_path_train + "/VisDrone2019-DET-train/images"
# Remove the target images folder if it exists
if os.path.exists(target_folder):
    shutil.rmtree(target_folder)
    print(f"Removed existing folder: {target_folder}")
# move the source images folder to the target location
shutil.move(source_folder, target_folder)
shutil.rmtree(extracted_folder_path_train)
print(f"Moved images from {source_folder} to {target_folder}")


print("Successfully got labels and images for VisDrone datasets.")