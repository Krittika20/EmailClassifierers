#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################
### your code here!  name your classifier object clf if you want the

#########KMeans#################

# t0 = time()

# clf = KMeans(n_clusters = 2, random_state = 0)
# clf = clf.fit(features_train)

# print("training time", round(time() - t0, 3), "seconds")

# t1 = time()

# pred = clf.predict(features_test)
# acc = accuracy_score(labels_test, pred)

# print("predicting time:", round(time() - t1, 3), "seconds")

# print("Accuracy: ", acc) 

#########Random Forest##############

# t0 = time()

# clf = RandomForestClassifier(max_depth = 3, random_state = 1)
# clf = clf.fit(features_train, labels_train)

# print("training time", round(time() - t0, 3), "seconds")

# t1 = time()

# pred = clf.predict(features_test)

# print("predicting time:", round(time() - t1, 3), "seconds")

# acc = accuracy_score(labels_test, pred)

# print("Accuracy: ", acc)

##########Adaboost###################

t0 = time()

clf = clf = AdaBoostClassifier(n_estimators=50, random_state=1)
clf = clf.fit(features_train, labels_train)

print("training time", round(time() - t0, 3), "seconds")

t1 = time()

pred = clf.predict(features_test)

print("predicting time:", round(time() - t1, 3), "seconds")

acc = accuracy_score(labels_test, pred)

print("Accuracy: ", acc)

### visualization code (prettyPicture) to show you the decision boundary
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
