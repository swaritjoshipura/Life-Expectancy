import numpy as np
from sklearn.svm import SVC

def run(B,X_subset,y_subset,C):
    n = len(X_subset)
    bs_err = np.zeros(B)
    for b in range(B):
        train_samples = list(np.random.randint(0,n,n))
        test_samples = list(set(range(n)) - set(train_samples))
        alg = SVC(C=C,kernel='linear')
        alg.fit(X_subset[train_samples], y_subset[train_samples])
        bs_err[b] = np.mean(y_subset[test_samples] != alg.predict(X_subset[test_samples]))
    err = np.mean(bs_err)
    return err