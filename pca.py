from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def pca_evaluation(classifier,data,name,evaltill=51,evalonly=None,verbose=False):
    csvstring = "n_components,Test score,Train score\n"
    for i in range(2,evaltill):
        if evalonly:
            i = evalonly
        pca = PCA(n_components=i)  # adjust yourself
        pca.fit(data['train'][0])
        global_pca = pca
        X_t_train = pca.transform(data['train'][0])
        X_t_test = pca.transform(data['test'][0])
        classifier.fit(X_t_train, data['train'][1])
        row = str(i)+","+str(classifier.score(X_t_test,data['test'][1])) +","+str(classifier.score(X_t_train,data['train'][1]))+"\n"
        csvstring+=row
        scores = (str(classifier.score(X_t_train,data['train'][1])),str(classifier.score(X_t_test,data['test'][1])))
        if verbose:
            print(row)
        if evalonly:
            break
    if not evalonly:
        f = open("pca_evaluations/"+name+".csv",mode="w")
        f.write(csvstring)
        f.close()
    print(name,csvstring)
    return (classifier,scores)

def get_pca(data,number):
    pca = PCA(n_components=number)  # adjust yourself
    pca.fit(data['train'][0])
    return pca

def lda_evaluation(classifier,data,name,evaltill=51,evalonly=None,verbose=False):
    csvstring = "n_components,Test score,Train score\n"
    for i in range(2,evaltill):
        if evalonly:
            i = evalonly
        pca = LinearDiscriminantAnalysis(n_components=i)  # adjust yourself
        pca.fit(data['train'][0],data['train'][1])
        X_t_train = pca.transform(data['train'][0])
        X_t_test = pca.transform(data['test'][0])
        classifier.fit(X_t_train, data['train'][1])
        row = str(i)+","+str(classifier.score(X_t_test,data['test'][1])) +","+str(classifier.score(X_t_train,data['train'][1]))+"\n"
        csvstring+=row
        if verbose:
            print(row)
        if evalonly:
            break
    if not evalonly:
        f = open("pca_evaluations/"+name+"LDA.csv",mode="w")
        f.write(csvstring)
        f.close()
    print(name,csvstring)

