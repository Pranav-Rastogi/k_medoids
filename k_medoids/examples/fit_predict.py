from k_medoids import K_medoids
import numpy as np 

clf = K_medoids(3)
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [10000, 10000]])
clf.fit(X)

prediction = clf.predict([[1,1],[2,3],[0.5,2]])

print("Prediction is: {0}".format(prediction))