from sklearn.neighbors import KNeighborsClassifier
from pca import pca_evaluation


def decide_parameters(data):
    for neighbours in range(2,20):
        # With PCA 40 features
        pca_evaluation(KNeighborsClassifier(n_neighbors=neighbours,weights="distance"),data=data,name="KNN"+str(neighbours),evalonly=40)
        # Without PCA
        neigh = KNeighborsClassifier(n_neighbors=neighbours,weights="distance")
        neigh.fit(data['train'][0],data['train'][1])
        print(neighbours,neigh.score(data['test'][0],data['test'][1]),neigh.score(data['train'][0],data['train'][1]))