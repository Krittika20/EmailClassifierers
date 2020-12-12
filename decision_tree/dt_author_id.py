#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
sys.path.append("ud120-projects/tools")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
t0 = time()

clf = DecisionTreeClassifier(min_samples_split = 40)

temp = clf.fit(features_train, labels_train)

print("training time", round(time() - t0, 3), "seconds")

t1 = time()
pred = temp.predict(features_test)

print("predicting time:", round(time() - t1, 3), "seconds")

acc = accuracy_score(labels_test, pred)
print("Accuracy: ", acc)

print("Number of features in the data", len(features_train[0]))
#########################################################


