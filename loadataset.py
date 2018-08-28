from PIL import Image
import numpy as np
from time import sleep
import scipy.io as sio
from skimage.transform import resize
from transforms import smart_image_resize
import scipy.io as sio
from sklearn.externals import joblib


def prepare_data(X,y,Xrow,yrow):
    X.append(Xrow)
    y.append(yrow)

def loading_dataset(X,y,new_shape=(15,15)):
    heights = []
    widths = []
    for file in range(1,51):
        if file < 10:
            name = 'nisttrain_cell/file_000'+str(file)+'.mat'
        else:
            name = 'nisttrain_cell/file_00'+str(file)+'.mat'
        mat_contents = sio.loadmat(name)
        for image in mat_contents['imcells'][0]:
            height,width = image.shape
            heights.append(height)
            widths.append(width)
            prepare_data(X,y,smart_image_resize(image,new_shape=new_shape).flatten(),int((file-1)/5))
    X = np.array(X)
    y = np.array(y)
    heights = np.array(heights)
    widths = np.array(widths)
    print("RANGE of HEIGHTS:", np.min(heights), np.max(heights))
    print("RANGE of WIDTHS:", np.min(widths), np.max(widths))
    joblib.dump((X,y),'nist'+str(new_shape[0])+'.pkl')
    print("Dumping:",'nist'+str(new_shape[0])+'.pkl')
    return (X,y)


def loading_image(filepath):
    img = Image.open(filepath)
    iar = np.asarray(img)
    X = smart_image_resize(iar,binary_encoded=False,plot=False)
    print(X)
    return X


