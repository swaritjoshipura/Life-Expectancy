import numpy as np
from sklearn.svm import SVC
import preprocessing
import pcalearn
import pcaproj

reduced_data, column_names = preprocessing.run("lifeexpecdata.csv")

positive_samples = list(np.where(column_names==1.)[0])
negative_samples = list(np.where(column_names<=0.)[0])

samples_in_fold1 = positive_samples[0:824] + negative_samples[0:824]
samples_in_fold2 = positive_samples[824:] + negative_samples[824:]

#print(samples_in_fold2)

F = 21
C = 2.0
gamma = 1000.0

y_pred = np.zeros(len(reduced_data),int)
X_fold1 = reduced_data[samples_in_fold1]
X_fold2 = reduced_data[samples_in_fold2]
mu_fold1, Z_fold1 = pcalearn.run(F, X_fold1)
X_fold1_small = pcaproj.run(X_fold1, mu_fold1, Z_fold1)
X_fold2_small = pcaproj.run(X_fold2, mu_fold1, Z_fold1)
alg = SVC(C=C,kernel='rbf',gamma=gamma)
alg.fit(X_fold1_small, column_names[samples_in_fold1])
y_pred[samples_in_fold2] = alg.predict(X_fold2_small)

mu_fold2, Z_fold2 = pcalearn.run(F, X_fold2)
X_fold1_small = pcaproj.run(X_fold1, mu_fold2, Z_fold2)
X_fold2_small = pcaproj.run(X_fold2, mu_fold2, Z_fold2)
alg = SVC(C=C,kernel='rbf',gamma=gamma)
alg.fit(X_fold2_small, column_names[samples_in_fold2])
y_pred[samples_in_fold1] = alg.predict(X_fold1_small)

print(np.mean(column_names!=y_pred))
