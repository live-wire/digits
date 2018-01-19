from PIL import Image
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from time import sleep
import random
import scipy.io as sio
from skimage.transform import resize
import scipy.io as sio
from sklearn.externals import joblib

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# local imports \/ \/
from loadataset import loading_dataset
from transforms import smart_image_resize
from svm import SupportVectorMachine
from settings import USE_TRAINED_MODELS
from pca import pca_evaluation
from pca import lda_evaluation
from myutils import backup_util
from scipy.stats import mode
from pca import get_pca
from hog import get_hog_features
from combining import average_of_classifiers
from combining import product_of_classifiers

# Chose parameters for svc,knn and mlp classifiers after fine-tuning
def classifiers_init():
    # classifier_labels = ["svc","knn","lda","qda","logistic","gaussian","mlp","randomforest"]
    # classifiers = [svm.SVC(probability=True),KNeighborsClassifier(n_neighbors=4,weights="distance"),LinearDiscriminantAnalysis(),
    #                    QuadraticDiscriminantAnalysis(),LogisticRegression(),GaussianNB(),
    #                    MLPClassifier(solver='lbfgs', max_iter=1000, learning_rate="adaptive", activation="tanh", alpha=10,
    #                                  hidden_layer_sizes=(20, 20), random_state=1),RandomForestClassifier()]
    classifier_labels = ["svc", "knn", "mlp"]
    classifiers = [svm.SVC(probability=True), KNeighborsClassifier(n_neighbors=4,weights="distance"),
                   MLPClassifier(solver='lbfgs', max_iter=1000, learning_rate="adaptive", activation="tanh", alpha=10,
                                 hidden_layer_sizes=(20, 20), random_state=1)]
    return (classifier_labels,classifiers)


def split_nist_dataset(X,y,train_size_per_class = 900,test_size_per_class = 100,number_of_samples_per_class=1000,number_of_classes=10):
    Xtrain,ytrain,Xtest,ytest = [],[],[],[]
    buffer_size = number_of_samples_per_class/train_size_per_class
    buffer = []
    decimal_collected = 0
    for i,row in enumerate(X):
        buffer.append(i)
        if len(buffer) + decimal_collected >= buffer_size:
            decimal_collected = len(buffer) + decimal_collected - buffer_size
            elem = random.choice(buffer)
            Xtrain.append(X[elem])
            ytrain.append(y[elem])
            for item in buffer:
                if not item==elem:
                    Xtest.append(X[item])
                    ytest.append(y[item])
            buffer = []
    return {'train':(np.array(Xtrain),np.array(ytrain)),'test':(np.array(Xtest),np.array(ytest))}


def try_all_classifiers(data, shape, train_size, n_feature, type="pca"):
    classifier_labels,classifiers = classifiers_init()
    rowtext = "\n"
    rowtext+=str(shape[0])+","+str(train_size)+","+str(n_feature)
    print("TRY - ALL - CLASSIFIERS("+type+") with: pixels,train_size,n_components",rowtext)
    evaluation = pca_evaluation
    clfs = []
    if type=="pca":
        evaluation = pca_evaluation

    for i,clfr in enumerate(classifiers):
        if not n_feature == "NA":
            clf,scores = evaluation(clfr, data=data, name=classifier_labels[i], evalonly=n_feature, verbose=True)
            clfs.append(clf)
            rowtext+=","+ scores[0]+"," + scores[1]
        else:
            print(classifier_labels[i])
            if not type=="HOG":
                clfr.fit(data['train'][0],data['train'][1])
                rowtext += "," + str(clfr.score(data['train'][0],data['train'][1])) + "," + \
                           str(clfr.score(data['test'][0],data['test'][1]))
            else:
                Xtrain = get_hog_features(data['train'][0],shape=shape)
                Xtest = get_hog_features(data['test'][0],shape=shape)
                clfr.fit(Xtrain, data['train'][1])

                rowtext += "," + str(clfr.score(Xtrain, data['train'][1])) + "," + str(
                    clfr.score(Xtest, data['test'][1]))

    if not n_feature == "NA":
        f = n_feature
        clfs = clfs
    else:
        f = None
        clfs = classifiers
    avg = average_of_classifiers(data,clfs, pca_number=f)
    prod = product_of_classifiers(data,clfs, pca_number=f)
    rowtext += "," + str(avg[1][0]) + "," + str(avg[1][1])
    rowtext += "," + str(prod[1][0]) + "," + str(prod[1][1])
    print(rowtext)
    if not n_feature == "NA":
        f = open("evaluation_scenario1_top3_with_"+type+".csv","a")
    else:
        if not type=="HOG":
            f = open("evaluation_scenario1_top3_pixels.csv","a")
        else:
            f = open("evaluation_with_"+type+"_and_pca.csv","a")
    f.write(rowtext)
    f.close()

print("Loading data...")
X=[]
y=[]
# These are the new shapes of the resized images for training our models and testing
try_shapes = [(15,15)]
train_size_per_class = [800]
n_features = [50]
# n_features = ["NA"]
for shape in try_shapes:
    for train_size in train_size_per_class:
        for feature in n_features:
            X = []
            y = []
            X,y = backup_util('nist'+str(shape[0])+'.pkl',loading_dataset,X,y,shape)
            print("Dataset shape:",X.shape,y.shape)
            print("Splitting data...")
            data = split_nist_dataset(X,y,train_size_per_class=train_size)
            print("Split: ",data['train'][0].shape,data['test'][0].shape)
            # PCA Evaluations
            # try_all_classifiers(data, shape, train_size, feature)
            # Dissimilarity Evaluations
            try_all_classifiers(data, shape, train_size, feature)