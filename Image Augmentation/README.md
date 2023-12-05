# Image Augmentation Using Albumentations
This Python script reads all images in a given directory and applies various pixel-level transformations from the Albumentations library to them. The resulting transformed images are written to a new directory created in the working directory of the script.

## Requirements
The script imports OpenCV and Albumentations, so these must be installed on the host machine to run it. The local directory containing the script must contain folders for the input images and annotations you wish to transform. By default these folders are "visdrone-images" and "visdrone-annotation" respectively. The output folders containing transformed images and annotations will automatically be created. By default these are "image-transformation-directory" and "annotation-transformation-directory" respectively. Any pointers to folders can be changed in the img_augmentation.py file to match the names of your current folders.

## Usage
To use, ensure the folders for your initial images and transformations match the names of the folders in the img_augmentation.py file as described in the requirements section. In a command line, simply run "python img_augmentation.py" using a python installation or virtual environment that has the OpenCV and Albumentations packages installed. The transformed images and annotations will be put into new folders specified by the output folder names in the python file.

## Output
The script will output two folders, one for transformed images, and one for transformed annotations. By default, these folders are named "image-transformation-directory" and "annotation-transformation-directory" respectively. The images will be transformed according to the conglomerate transformation model, with one or more image transformations being applied to each output image. The names of the files are dynamically changed to match between the image and annotation directories, so the transformed folders can be directly for model training.
