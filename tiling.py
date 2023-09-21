#############################################################################################
# Filename: tiling.py
# Author: Hayley Wisman
# Date Created: 04/06/2023
# Last Modified: 09/21/2023
# Description: Performs tiling on an image as a preprocessing step prior to object detection
#############################################################################################


import cv2
import numpy as np
import math
import sys

def tiling(img):
    # read and display an image in the same directory to test output
    img = cv2.imread(img)

    # set configurable variables below
    x_tiles = 4  # number of tiles in x-direction
    y_tiles = 4  # number of tiles in y-direction
    resize_factor = 2  # factor by which image tile will be upsized in x and y directions
    # set variables here to specify tile size instead of # tiles

    ####################################################
    # Edited By: Jalen Neal
    # dynamically calculates tiles in x and y direction
    for i in range(img.shape[0]):
        if i >= 4 and (img.shape[0] % i == 0):
            x_tiles = min(i, (img.shape[0] // i))
            break

    for i in range(img.shape[1]):
        if i >= 4 and (img.shape[1] % i == 0):
            y_tiles = min(i, (img.shape[1] // i))
            break

    ################################################
    # Compute pixel dimensions per tile
    xtile_pixels = int(img.shape[0] / x_tiles)
    ytile_pixels = int(img.shape[1] / y_tiles)

    resize_dim = (resize_factor * ytile_pixels, resize_factor * xtile_pixels)  # resized tile dimensions
    tiles = []  # stores tiles after they have been resized

    # store tiles in an array
    for i in range(x_tiles):
        startx = i * xtile_pixels
        endx = (i + 1) * xtile_pixels

        for j in range(y_tiles):
            starty = j * ytile_pixels
            endy = (j + 1) * ytile_pixels
            tile = cv2.resize(img[startx:endx, starty:endy], resize_dim, interpolation=cv2.INTER_CUBIC)
            tiles.append(tile)

    return tiles


tiles = tiling("eye1.jpg")
# display full image and a tile (testing/debugging)

cv2.imshow('image1', cv2.imread("eye1.jpg"))
for i, t in enumerate(tiles):
    title = 'resized tile{}'.format(i)
    cv2.imshow(title, tiles[i])

cv2.waitKey(0)  # this keeps the image windows open until the user hits the 0 key

