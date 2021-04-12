import preprocessing
import numpy as np

reduced_data, column_names = preprocessing.run("lifeexpecdata.csv")

print(reduced_data)
print(column_names)

#np.savetxt("foo.csv", reduced_data, delimiter=",")