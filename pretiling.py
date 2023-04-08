#############################################################################################
# Filename: pretiling.py
# Author: Hayley Wisman
# Date Created: 04/06/2023
# Last Modified: 04/06/2023
# Description: Performs tiling on an image as a preprocessing step prior to object detection
#############################################################################################

import cv2
import numpy as np

# read and display an image to test output
img = cv2.imread("test_img.jpg")
x_tiles = 5
y_tiles = 6

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

print(x_tiles)    # just for debugging
print(y_tiles)

# Compute pixel dimensions per tile
xtile_pixels = int(img.shape[0] / x_tiles)
ytile_pixels = int(img.shape[1] / y_tiles)
tiles = []

# store tiles in an array
for i in range(x_tiles):
    startx = i * xtile_pixels     # will leave 1 px overlap--needs to be adjusted
    endx = (i + 1) * xtile_pixels

    for j in range(y_tiles):
        starty = j * ytile_pixels
        endy = (j + 1) * ytile_pixels
        tiles.append(img[startx:endx, starty:endy])

# display full image and a tile (testing/debugging)
cv2.imshow('image', img)
cv2.imshow('tile0', tiles[0])

cv2.waitKey(0)     # this keeps the image windows open until the user hits 0 key

# use cubic interpolation to upsize the tiles

# Note visdrone is more representative of certain cities
# don't necessarily need to run different models at the same number of epochs
# just use early exit condition based on change
# random dropout?