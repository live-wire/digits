from sklearn.neural_network import MLPClassifier
from pca import pca_evaluation
def decide_parameters(data):
    alphas = [1e-5,1e-3,1e-1,1,10,100,1000]
    for alp in alphas:
        print("Alpha=",alp)
        clf = MLPClassifier(solver='lbfgs',max_iter=1000,learning_rate="adaptive",activation="tanh", alpha=alp,hidden_layer_sizes=(20,20), random_state=1)
        pca_evaluation(clf,data=data,name="mlp",verbose=True,evalonly=42)
    # pca_evaluation(clf,data=data,name="mlp",verbose=True,evalonly=40)