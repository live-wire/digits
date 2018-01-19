from PIL import Image
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from functools import reduce
import scipy.misc



# Converts the image to black and white
def black_white(img,image=False):
    new = []
    for i,row in enumerate(img):
        if image:
            temp = [0 if np.array(x).min()<=180 else 255 for x in row]
        else:
            temp = [0 if x<=100 else 255 for x in row]
        new.append(temp)
    return np.array(new)

# Binary encodes the image
def binary_image(img):
    new = []
    for i,row in enumerate(img):
        temp = [1 if x<255 else 0 for x in row]
        new.append(temp)
    return np.array(new)

#Reduces image to 15x15
def smart_image_resize(img,new_shape = (15,15),binary_encoded=True,plot=False):
    new = []
    if binary_encoded:
        for i,row in enumerate(img):
            temp = [0 if x == 1 else 255 for x in row]
            new.append(temp)

        new = np.array(new)
    else:
        new = img
    small = scipy.misc.imresize(new, new_shape)
    small = black_white(small,image=True)

    if plot:
        ax1 = plt.subplot2grid((8, 7), (0, 0), rowspan=8, colspan=3)
        ax2 = plt.subplot2grid((8, 7), (0, 4), rowspan=8, colspan=3)
        ax1.imshow(img)
        ax2.imshow(small)
        plt.pause(0.2)


    return binary_image(small)

def reverse_image(image):
    new = []
    for i, row in enumerate(image):
        temp = [255-x for x in row]
        new.append(temp)
    return np.array(new)

def image_resize(img,new_shape=(15,15),binary_encoded=True,reverse=False):
    new = []
    if binary_encoded:
        for i, row in enumerate(img):
            temp = [0 if x == 1 else 255 for x in row]
            new.append(temp)

        new = np.array(new)
    else:
        new = img
    small = scipy.misc.imresize(new, new_shape)
    small = black_white(small, image=True)
    if reverse:
        return reverse_image(small)
    else:
        return small