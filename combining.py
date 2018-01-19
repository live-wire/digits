from PIL import Image
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from neuralnet import predictcnn
from pca import get_pca
from scipy.stats import mode

def voting_classifier(trained_clfs,data,plot=False):

    Xtest = get_pca(data,35).transform(data['test'][0])
    predictions_cnn = predictcnn(data['test'][0].reshape(-1,15,15,1))
    ytest = data['test'][1]
    result_features = []
    misclassified = 0
    for i,row in enumerate(Xtest):
        temp = []
        for clf in trained_clfs:
            temp2 = []
            temp2.append(row)
            result = clf.predict(np.array(temp2))
            temp.append(result)
        tempcnn = []
        tempcnn.append(predictions_cnn[i])
        temp.append(tempcnn)
        most_frequent = mode(temp)[0][0]
        if not most_frequent == ytest[i]:
            print(temp,ytest[i])
            misclassified = misclassified+1
            if plot:
                ax1 = plt.subplot2grid((8, 7), (0, 0), rowspan=8, colspan=3)
                # ax2 = plt.subplot2grid((8, 7), (0, 4), rowspan=8, colspan=3)
                print(data['test'][0][i].reshape(15,15),most_frequent,"Should have been:",data['test'][1][i])
                ax1.imshow(data['test'][0][i].reshape(15,15))
                # ax1.title = most_frequent
                # ax2.imshow(small)
                plt.pause(2)

                # plt.pause(2)
        temp.append(most_frequent)
        result_features.append(temp)
    result_features = np.array(result_features)
    print("Accuracy = ",(len(ytest)-misclassified)/len(ytest))



def average_of_classifiers(data,clfrs,pca_number = None):
    print("AVG Using " + str(len(clfrs)) + " Classifiers")
    trainerr = 0
    testerr = 0
    yreturn = []
    if pca_number == None:
        Xtest = data['test'][0]
        Xtrain = data['train'][0]
    else:
        pca_x = get_pca(data,pca_number)
        Xtest = pca_x.transform(data['test'][0])
        Xtrain = pca_x.transform(data['train'][0])
        pca_x = None
    for i,row in enumerate(Xtrain):
        actual = data['train'][1][i]
        sum = []
        for clfr in clfrs:
            if len(sum)==0:
                sum = np.zeros(clfr.predict_proba(row.reshape(1,-1)).shape)
            sum += clfr.predict_proba(row.reshape(1,-1))
        ind = (-sum).argsort()[:1][0][0]
        yreturn.append(ind)
        if not ind==actual:
            # print("Avg predicted=",ind,"----Actual=",actual)
            trainerr+=1
    print("TrainError=", str(trainerr / len(data['test'][1])))
    for i,row in enumerate(Xtest):
        actual = data['test'][1][i]
        sum=[]
        for clfr in clfrs:
            if len(sum)==0:
                sum = np.zeros(clfr.predict_proba(row.reshape(1,-1)).shape)
            sum += clfr.predict_proba(row.reshape(1,-1))
        ind = (-sum).argsort()[:1][0][0]
        yreturn.append(ind)
        if not ind==actual:
            # print("Avg predicted=",ind,"----Actual=",actual)
            testerr+=1
    print("TestError=",str(testerr/len(data['test'][1])))

    scoreTrain = 1-(trainerr/len(data['train'][1]))
    scoreTest = 1-(testerr / len(data['test'][1]))
    return (np.array(yreturn),(scoreTrain,scoreTest))


def product_of_classifiers(data,clfrs, pca_number = None):
    print("Product Using " + str(len(clfrs)) + " Classifiers")
    trainerr = 0
    testerr = 0
    yreturn = []
    if pca_number == None:
        Xtest = data['test'][0]
        Xtrain = data['train'][0]
    else:
        pca_x = get_pca(data, pca_number)
        Xtest = pca_x.transform(data['test'][0])
        Xtrain = pca_x.transform(data['train'][0])
        pca_x = None
    for i, row in enumerate(Xtrain):
        actual = data['train'][1][i]
        prod = []
        for clfr in clfrs:
            if len(prod) == 0:
                prod = np.ones(clfr.predict_proba(row.reshape(1, -1)).shape)
            prod *= clfr.predict_proba(row.reshape(1, -1))
        ind = (-prod).argsort()[:1][0][0]
        yreturn.append(ind)
        if not ind == actual:
            # print("Prod predicted=", ind, "----Actual=", actual)
            trainerr += 1
    print("TrainError=", str(trainerr / len(data['test'][1])))
    for i, row in enumerate(Xtest):
        actual = data['test'][1][i]
        prod = []
        for clfr in clfrs:
            if len(prod) == 0:
                prod = np.ones(clfr.predict_proba(row.reshape(1, -1)).shape)
            prod *= clfr.predict_proba(row.reshape(1, -1))
        ind = (-prod).argsort()[:1][0][0]
        yreturn.append(ind)
        if not ind == actual:
            # print("Prod predicted=", ind, "----Actual=", actual)
            testerr += 1
    print("TestError=", str(testerr / len(data['test'][1])))

    scoreTrain = 1 - (trainerr / len(data['train'][1]))
    scoreTest = 1 - (testerr / len(data['test'][1]))
    return (np.array(yreturn), (scoreTrain, scoreTest))


def stacking_classifiers(data,clfrs, pca_number = None):
    print("Product Using " + str(len(clfrs)) + " Classifiers")
    trainerr = 0
    testerr = 0
    yreturn = []
    if pca_number == None:
        Xtest = data['test'][0]
        Xtrain = data['train'][0]
    else:
        pca_x = get_pca(data, pca_number)
        Xtest = pca_x.transform(data['test'][0])
        Xtrain = pca_x.transform(data['train'][0])
        pca_x = None
    for i, row in enumerate(Xtrain):
        actual = data['train'][1][i]
        prod = []
        for clfr in clfrs:
            if len(prod) == 0:
                prod = np.ones(clfr.predict_proba(row.reshape(1, -1)).shape)
            prod *= clfr.predict_proba(row.reshape(1, -1))
        ind = (-prod).argsort()[:1][0][0]
        yreturn.append(ind)
        if not ind == actual:
            # print("Prod predicted=", ind, "----Actual=", actual)
            trainerr += 1
    print("TrainError=", str(trainerr / len(data['test'][1])))
    for i, row in enumerate(Xtest):
        actual = data['test'][1][i]
        prod = []
        for clfr in clfrs:
            if len(prod) == 0:
                prod = np.ones(clfr.predict_proba(row.reshape(1, -1)).shape)
            prod *= clfr.predict_proba(row.reshape(1, -1))
        ind = (-prod).argsort()[:1][0][0]
        yreturn.append(ind)
        if not ind == actual:
            # print("Prod predicted=", ind, "----Actual=", actual)
            testerr += 1
    print("TestError=", str(testerr / len(data['test'][1])))

    scoreTrain = 1 - (trainerr / len(data['train'][1]))
    scoreTest = 1 - (testerr / len(data['test'][1]))
    return (np.array(yreturn), (scoreTrain, scoreTest))

