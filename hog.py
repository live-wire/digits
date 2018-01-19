import cv2
from PIL import Image
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.misc import imresize
from transforms import image_resize
from time import sleep
from sklearn import preprocessing
from pca import get_pca

def get_hog_features(X,shape=(15,15)):
    winSize = shape
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True
    useSignedGradients = 1

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType,
                            L2HysThreshold, gammaCorrection, nlevels, useSignedGradients)
    new_x = []
    for item in X:
        x = item.reshape(shape)
        x = image_resize(x,new_shape=shape,binary_encoded=True,reverse=True)
        x = np.asarray(Image.fromarray(x,'L'))
        temp = hog.compute(x)
        temp = np.array(temp)
        # print(temp.flatten())
        new_x.append(temp.flatten())

    X_normalized = preprocessing.normalize(np.array(new_x), norm='l2')
    # return X_normalized
    # trying Hog with PCA of pixels
    pca_prep = {'train':(X,False)}
    pca_result = get_pca(pca_prep,40)
    x_pca = pca_result.transform(X)
    print(X_normalized.shape,X.shape,x_pca.shape)
    Z = np.hstack((x_pca,X_normalized))
    print(Z.shape)
    return Z


