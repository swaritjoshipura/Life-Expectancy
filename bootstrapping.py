import numpy as np
from sklearn.svm import SVC


def bootstrapping(B,X_subset,y_subset,C):
    n = len(X_subset)
    bs_err = np.zeros(B)
    for b in range(B):
        train_samples = list(np.random.randint(0,n,n))
        test_samples = list(set(range(n)) - set(train_samples))
        alg = SVC(C=C,kernel='rbf')
        alg.fit(X_subset[train_samples], y_subset[train_samples])
        bs_err[b] = np.mean(y_subset[test_samples] != alg.predict(X_subset[test_samples]))
    err = np.mean(bs_err)
    return err

def run(data, labels):
    positive_samples = np.where(labels == 1)[0]
    negative_samples = np.where(labels == -1)[0]
    zero_samples = np.where(labels == 0)[0]

    train_samples = list(positive_samples[0:int(np.floor(len(positive_samples)/3))]) + list(negative_samples[0:int(np.floor(len(negative_samples)/3))]) + list(zero_samples[0:int(np.floor(len(zero_samples)/3))])  

    validation_samples = list(positive_samples[int(np.floor(len(positive_samples)/3)):int(np.floor(2*len(positive_samples)/3))]) + list(negative_samples[int(np.floor(len(negative_samples)/3)):int(np.floor(2*len(negative_samples)/3))]) + list(zero_samples[int(np.floor(len(zero_samples)/3)):int(np.floor(2*len(zero_samples)/3))])
    

    test_samples = list(positive_samples[int(np.floor(2*len(positive_samples)/3)):len(positive_samples)]) + list(negative_samples[int(np.floor(2*len(negative_samples)/3)):len(negative_samples)]) + list(zero_samples[int(np.floor(2*len(zero_samples)/3)):len(zero_samples)])


    C_list = [0.1, 1.0, 10.0]
    B = 30
    
    labels_pred = np.zeros(len(data),int)
    best_err = 1.1 # Any value greater than 1
    best_C = 0.0
    for C in C_list:
        err = bootstrapping(B,data[train_samples], labels[train_samples],C)
        print ("C=", C, ", err=", err)
        if err < best_err:
            best_err = err
            best_C = C
    print ("best_C=", best_C)
    

    alg = SVC(C=best_C,kernel='rbf')
    alg.fit(data[train_samples], labels[train_samples])
    
    labels_pred[validation_samples] = alg.predict(data[validation_samples])
    best_err = 1.1 # Any value greater than 1
    best_C = 0.0
    for C in C_list:
        err = bootstrapping(B,data[validation_samples], labels[validation_samples],C)
        print ("C=", C, ", err=", err)
        if err < best_err:
            best_err = err
            best_C = C
    print ("best_C=", best_C)

    
    alg = SVC(C=best_C,kernel='rbf')
    alg.fit(data[train_samples], labels[train_samples])
    labels_pred = alg.predict(data[test_samples])
    err = np.mean(labels[test_samples] != np.array([labels_pred]).T)

    print("final best err = " + str(err))

    return err