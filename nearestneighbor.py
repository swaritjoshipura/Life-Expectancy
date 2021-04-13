import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import preprocessing

reduced_data, column_names = preprocessing.run("lifeexpecdata.csv")

positive_samples = list(np.where(column_names==1.)[0])
negative_samples = list(np.where(column_names<=0.)[0])

samples_in_fold1 = positive_samples[0:250] + negative_samples[0:250]
samples_in_fold2 = positive_samples[250:] + negative_samples[250:]

k = 50

alg = KNeighborsClassifier(n_neighbors=k,algorithm='brute')
alg.fit(reduced_data,column_names)
y_pred = alg.predict(reduced_data)

err = np.mean(column_names != np.array([y_pred]).T)

print(err)