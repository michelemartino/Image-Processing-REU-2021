"""
Image Processing
Code session 2 Tripods NSF REU-Graduate Stem for All 2021
"""

# Imports image
from PIL import Image
import numpy as np
import math as math

# turn image into a 3-D matrix
image = Image.open('Totti.jpg')
data = np.asarray(image)
image.show()

# makes image grayscale
data_gray = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]
gray_image = Image.fromarray(data_gray)
gray_image.show()

# defines convoluting function and its parameters
def convolute(kernel, data, pad_width):
    # pads image data with 0s to prepare for convolution
    data_padded = np.pad(data, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode = 'constant')
    output = data.copy()
    # convolutes data image with kernel filer matrix
    for x in range(0, 2):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                output[i, j, x] = np.sum(data_padded[i : i + kernel.shape[0], j : j + kernel.shape[0], x] * kernel[:, :])
    return(output)

# takes image data as input and then shows image    
def printImage(data):
    data = data.astype(np.uint8)
    data_image = Image.fromarray(data)
    data_image.show()

# sets up blur filter, convulutes and prints blurred image
n = 15
box_filter = np.full((n,n), 1/(n*n))
data_blurred = convolute(box_filter, data, pad_width = math.floor(n/2))
printImage(data_blurred)

#sharpens image using mask which is made from original and blurred image
mask = data - data_blurred
data_sharp = data + mask
printImage(data_sharp)

# defines vertical laplacian operator filters and convulutes ina different formula to get edged image
def edge(data):
    edge_filter_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    edge_filter_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    output = np.sqrt((convolute(edge_filter_x, data, pad_width = 1))**2 + (convolute(edge_filter_x, data, pad_width = 1))**2)
    return(output)

# copes with the fact that resulting image might be too dark
def threshold(data, t):
    output = data.copy()
    for x in range(0, 2):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if(data[i, j, x] < t):
                    output[i, j, x] = 0
                else:
                    output[i, j, 0] = 0
                    output[i, j, 1] = 255
                    output[i, j, 2] = 0
    return(output)

# gets edged image, adjust darkness problem with trashold function and prints
data_edge = edge(data_blurred)
data_edge2 = threshold(data_edge, 15)
printImage(data_edge2)



