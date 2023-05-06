#############################################################################################
# Filename: tiling.py
# Author: Hayley Wisman
# Date Created: 04/06/2023
# Last Modified: 04/06/2023
# Description: Performs tiling on an image as a preprocessing step prior to object detection
#############################################################################################
import cv2
import numpy as np
import math
import sys

# read and display an image in the same directory to test output
img = cv2.imread("test_img.jpg")

# set configurable variables below
x_tiles = 4             # number of tiles in x-direction
y_tiles = 4             # number of tiles in y-direction
resize_factor = 2       # factor by which image tile will be upsized in x and y directions
# set variables here to specify tile size instead of # tiles

# Compute pixel dimensions per tile
xtile_pixels = int(img.shape[0] / x_tiles)
ytile_pixels = int(img.shape[1] / y_tiles)

# Ensure number of tiles evenly divides image dimensions
# range is preset arbitrarily--adjust as desired
for i in range(5):
    if (img.shape[0] % x_tiles == 0):
        break
    else:
        x_tiles += 1

for i in range(5):
    if (img.shape[1] % y_tiles == 0):
        break
    else:
        y_tiles += 1

print("Tiles in x-direction: ", x_tiles)    # just for debugging
print("Tiles in y-direction: ", y_tiles)

resize_dim = (resize_factor*ytile_pixels, resize_factor*xtile_pixels)    # resized tile dimensions
tiles = []         # stores tiles after they have been resized

# store tiles in an array
for i in range(x_tiles):
    startx = i * xtile_pixels
    endx = (i + 1) * xtile_pixels

    for j in range(y_tiles):
        starty = j * ytile_pixels
        endy = (j + 1) * ytile_pixels
        tile = cv2.resize(img[startx:endx, starty:endy], resize_dim, interpolation = cv2.INTER_CUBIC)
        tiles.append(tile)

# display full image and a tile (testing/debugging)
cv2.imshow('image1', img)
cv2.imshow('resized tile0', tiles[0])
#cv2.imshow('resized tile1', tiles[1])
#cv2.imshow('resized tile2', tiles[2])
#cv2.imshow('resized tile3', tiles[3])

cv2.waitKey(0)     # this keeps the image windows open until the user hits the 0 key