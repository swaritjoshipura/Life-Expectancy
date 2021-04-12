import numpy as np
from sklearn.svm import SVC
import bootstrapping
import preprocessing
import pcalearn
import pcaproj

reduced_data, column_names = preprocessing.run("lifeexpecdata.csv")

positive_samples = list(np.where(column_names==1.)[0])
negative_samples = list(np.where(column_names<=0.)[0])

samples_in_fold1 = positive_samples[0:250] + negative_samples[0:250]
samples_in_fold2 = positive_samples[250:] + negative_samples[250:]

C_list = [0.1, 1.0, 10.0]
B = 22
y_pred = np.zeros(len(reduced_data),int)
best_err = 1.1
best_C = 0.0
for C in C_list:
    err = bootstrapping.run(B,reduced_data[samples_in_fold1], column_names[samples_in_fold1],C)
    print("C=", C, ", err=", err)
    if err < best_err:
        best_err = err
        best_C = C
        print("best_C=", best_C)

alg = SVC(C=best_C,kernel='linear')
alg.fit(reduced_data[samples_in_fold1], column_names[samples_in_fold1])
y_pred[samples_in_fold2] = alg.predict(reduced_data[samples_in_fold2])
best_err = 1.1
best_C = 0.0
for C in C_list:
    err = bootstrapping.run(B,reduced_data[samples_in_fold2], column_names[samples_in_fold2],C)
    print("C=", C, ", err=", err)
    if err < best_err:
        best_err = err
        best_C = C
print("best_C=", best_C)

alg = SVC(C=best_C,kernel='linear')
alg.fit(reduced_data[samples_in_fold2], column_names[samples_in_fold2])
y_pred[samples_in_fold1] = alg.predict(reduced_data[samples_in_fold1])
err = np.mean(column_names!=y_pred)
print(err)
