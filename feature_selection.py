# Trying feature selection

from PIL import Image
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn import svm

def try_feature_selection(data):
    rfe = RFE(LinearSVC(), 50)
    rfe = rfe.fit(data['test'][0], data['test'][1])
    # print summaries for the selection of attributes
    print(rfe.support_)
    newx = []
    newxtest = []
    features_to_pick = []
    for ind,val in enumerate(rfe.support_):
        if val==True:
            features_to_pick.append(ind)
    for row in data['train'][0]:
        temp = []
        for val in features_to_pick:
            temp.append(row[val])
        newx.append(temp)
    newx = np.array(newx)
    for row in data['test'][0]:
        temp = []
        for val in features_to_pick:
            temp.append(row[val])
        newxtest.append(temp)
    newxtest = np.array(newxtest)
    svc = svm.SVC()
    svc.fit(newx,data['train'][1])
    print("RFE SCORE: ",svc.score(newxtest,data['test'][1]))
    print(rfe.ranking_)
    ranking = rfe.ranking_.reshape((15,15))
    # Plot pixel ranking
    plt.matshow(ranking, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Ranking of pixels with RFE")
    plt.show()