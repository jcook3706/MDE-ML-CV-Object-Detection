#############################################################################################
# Filename: img_augmentation.py
# Author: Hayley Wisman
# Date Created: 04/11/2023
# Last Modified: 04/11/2023
# Description: Uses the Albumentations library to perform image augmentation for the purpose
# of testing object detection model robustness across varying environmental conditions
#############################################################################################
import os
import cv2
import albumentations as A

# Directory of images to process -- set to VisDrone validation dataset by default
#read_directory = 'VisDrone2019-DET-val'
read_directory = 'testing'

# Create a new directory to store augmented images
# Will be created inside the current directory unless otherwise specified
write_directory = 'Visdrone-Robust-set'
os.mkdir(write_directory)

# read and augment every image in the given directory and write to output directory
for filename in os.listdir(read_directory):
    f = os.path.join(read_directory, filename)
    save_file = os.path.join(write_directory, filename)
    file_base = os.path.splitext(save_file)[0]
    file_ext = os.path.splitext(save_file)[1]

    if os.path.isfile(f):
        img = cv2.imread(f)

        # Blur_limit sets an upper bound for the randomized kernel size
        # Must be an odd number. Higher value = more potential blur
        motionblur = A.Compose([A.MotionBlur(always_apply=True, blur_limit=27)])
        rain = A.Compose([A.RandomRain(always_apply=True)])
        fog = A.Compose([A.RandomFog(always_apply=True)])
        defocus = A.Compose([A.Defocus(always_apply=True)])
        spatter = A.Compose([A.Spatter(always_apply=True, mode="mud")])
        transform_list = [motionblur, rain, fog, defocus, spatter]

        # OpenCV uses BGR color scheme -- convert to RGB for processing with Albumentations
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        i = 0  # concatenate with image file name to avoid overwriting the same image

        # apply each transform to the image
        for transform in transform_list:
            trans_img = transform(image=rgb_img)['image']
            # convert back to BGR to write image using OpenCV function
            trans_img = cv2.cvtColor(trans_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_base + str(i) + file_ext, trans_img)
            i += 1
