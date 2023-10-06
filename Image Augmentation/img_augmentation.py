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
import shutil

# Directory of images to process -- set to VisDrone validation dataset by default
read_directory = 'visdrone-images'
#read_directory = 'testing'    # this line for testing purposes only

# Create a new directory to store augmented images
# Will be created inside the current directory unless otherwise specified
single_transformation_write_directory = 'single-transformation-directory'
conglomerate_transformation_write_directory = 'conglomerate-transformation-directory'

if(os.path.exists(single_transformation_write_directory)):
    shutil.rmtree(single_transformation_write_directory)
if(os.path.exists(conglomerate_transformation_write_directory)):
    shutil.rmtree(conglomerate_transformation_write_directory)
os.mkdir(single_transformation_write_directory)
os.mkdir(conglomerate_transformation_write_directory)

# read and augment every image in the given directory and write to output directory
for filename in os.listdir(read_directory):
    f = os.path.join(read_directory, filename)
    single_save_file = os.path.join(single_transformation_write_directory, filename)
    single_file_base = os.path.splitext(single_save_file)[0]
    single_file_ext = os.path.splitext(single_save_file)[1]

    conglomerate_save_file = os.path.join(conglomerate_transformation_write_directory, filename)
    conglomerate_file_base = os.path.splitext(conglomerate_save_file)[0]
    conglomerate_file_ext = os.path.splitext(conglomerate_save_file)[1]

    if os.path.isfile(f):
        img = cv2.imread(f)

        # Blur_limit sets an upper bound for the randomized kernel size
        # Must be an odd number. Higher value = more potential blur
        motionblur = A.Compose([A.MotionBlur(always_apply=True, blur_limit=55)])
        rain = A.Compose([A.RandomRain(always_apply=True)])
        fog = A.Compose([A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, always_apply=True)])
        defocus = A.Compose([A.Defocus(always_apply=True)])
        spatter_rain = A.Compose([A.Spatter(always_apply=True)])
        spatter_mud = A.Compose([A.Spatter(mean=0.65, std=0.3, gauss_sigma=2, intensity=0.6, always_apply=True, mode='mud')])
        gaussian_noise = A.Compose([A.GaussNoise(always_apply=True)])
        transform_list = [motionblur, rain, fog, defocus, spatter_rain, spatter_mud, gaussian_noise]

        totalTransform = A.Compose([
            A.MotionBlur(p=0.5, blur_limit=35),
            A.RandomRain(p=0.5),
            A.RandomFog(p=0.5, fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1),
            A.Defocus(p=0.5),
            A.Spatter(p=0.5, mean=0.65, std=0.3, gauss_sigma=2, intensity=0.6, mode='mud'),
            A.Spatter(p=0.5, mean=0.65, std=0.3, gauss_sigma=2, intensity=0.6)
        ])

        # OpenCV uses BGR color scheme -- convert to RGB for processing with Albumentations
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        i = 0  # concatenate with image file name to avoid overwriting the same image

        # apply each transform to the image
        for transform in transform_list:
            trans_img = transform(image=rgb_img)['image']
            # convert back to BGR to write image using OpenCV function
            trans_img = cv2.cvtColor(trans_img, cv2.COLOR_RGB2BGR)
            if transform == motionblur:
                label = 'motionblur'
            elif transform == rain:
                label = 'rain'
            elif transform == fog:
                label = 'fog'
            elif transform == defocus:
                label = 'defocus'
            elif transform == spatter_mud:
                label = 'spattermud'
            elif transform ==  spatter_rain:
                label = 'spatterrain'
            elif transform == gaussian_noise:
                label = 'gaussiannoise'
            else:
                label = ''
            cv2.imwrite(single_file_base + '_' + label + '_' + single_file_ext, trans_img)
            i += 1

        k=0
        numTotalTransformImages = 5
        for i in range(numTotalTransformImages):
            trans_img = totalTransform(image=rgb_img)['image']
            trans_img = cv2.cvtColor(trans_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(conglomerate_file_base + 'total' + str(k) + conglomerate_file_ext, trans_img)
            k += 1