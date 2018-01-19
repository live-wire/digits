from sklearn import svm
from sklearn.externals import joblib
from settings import USE_TRAINED_MODELS
from sklearn import grid_search
from sklearn.model_selection import GridSearchCV

class SupportVectorMachine:
    def __init__(self,data):
        # expects data as a dictionary split into 'train' and 'test'
        self.data = data

    def svc_param_selection(self,X,y,nfolds):
        data  = self.data
        Cs = [0.001, 0.01, 0.1, 1, 10, 100]
        gammas = [0.001, 0.01, 0.1, 1, 10]
        param_grid = {'C': Cs, 'gamma': gammas}
        grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
        grid_search.fit(X, y)
        print(grid_search.best_params_)
        return grid_search.best_params_

    def train_rbf(self,C=1,gamma=0.01):
        data = self.data
        # SVC rbf kernel
        try:
            if not USE_TRAINED_MODELS:
                raise Exception("Don't use trained model!")
            svcrbf = joblib.load('trained/svcrbfc'+str(C)+'.pkl')
        except:
            svcrbf = svm.SVC(C=C)
            svcrbf.fit(data['train'][0], data['train'][1])
            joblib.dump(svcrbf, "trained/svcrbf"+str(C)+".pkl")
        print("SVM rbf C="+str(C)+" Test Score: ", svcrbf.score(data['test'][0], data['test'][1]), "\nTrain Score: ",
              svcrbf.score(data['train'][0], data['train'][1]))

    def train_all(self):
        data = self.data
        # SVC Linear kernel
        try:
            if not USE_TRAINED_MODELS:
                raise Exception("Don't use trained model!")
            svclinear = joblib.load('trained/svclinear.pkl')
        except:
            svclinear = svm.SVC(kernel="linear")
            svclinear.fit(data['train'][0], data['train'][1])
            joblib.dump(svclinear, "trained/svclinear.pkl")
        print("SVM linear Test Score: ", svclinear.score(data['test'][0], data['test'][1]), "\nTrain Score: ",
              svclinear.score(data['train'][0], data['train'][1]))

        # SVC poly kernel degree 3
        try:
            if not USE_TRAINED_MODELS:
                raise Exception("Don't use trained model!")
            svcpoly3 = joblib.load('trained/svcpoly3.pkl')
        except:
            svcpoly3 = svm.SVC(kernel="poly", degree=3)
            svcpoly3.fit(data['train'][0], data['train'][1])
            joblib.dump(svcpoly3, "trained/svcpoly3.pkl")
        print("SVM poly3 Test Score: ", svcpoly3.score(data['test'][0], data['test'][1]), "\nTrain Score: ",
              svcpoly3.score(data['train'][0], data['train'][1]))
        # SVC poly kernel degree 5
        try:
            if not USE_TRAINED_MODELS:
                raise Exception("Don't use trained model!")
            svcpoly4 = joblib.load('trained/svcpoly4.pkl')
        except:
            svcpoly4 = svm.SVC(kernel="poly", degree=5)
            svcpoly4.fit(data['train'][0], data['train'][1])
            joblib.dump(svcpoly4, "trained/svcpoly4.pkl")
        print("SVM poly4 Test Score: ", svcpoly4.score(data['test'][0], data['test'][1]), "\nTrain Score: ",
              svcpoly4.score(data['train'][0], data['train'][1]))

        # SVC rbf kernel
        try:
            if not USE_TRAINED_MODELS:
                raise Exception("Don't use trained model!")
            svcrbf = joblib.load('trained/svcrbf.pkl')
        except:
            svcrbf = svm.SVC()
            svcrbf.fit(data['train'][0], data['train'][1])
            joblib.dump(svcrbf, "trained/svcrbf.pkl")
        print("SVM rbf Test Score: ", svcrbf.score(data['test'][0], data['test'][1]), "\nTrain Score: ",
              svcrbf.score(data['train'][0], data['train'][1]))


