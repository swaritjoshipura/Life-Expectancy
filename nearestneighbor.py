import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import preprocessing

reduced_data, column_names = preprocessing.run("lifeexpecdata.csv")

k = 50

alg = KNeighborsClassifier(n_neighbors=k,algorithm='brute')
alg.fit(reduced_data,column_names)
y_pred = alg.predict(reduced_data)

err = np.mean(column_names != np.array([y_pred]).T)

print(err)