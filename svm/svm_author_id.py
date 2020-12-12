#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("ud120-projects/tools")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#########################################################
### your code goes here ###
t0 = time()
clf = SVC(kernel='rbf', C=10000.0)
temp = clf.fit(features_train, labels_train)
print("training time", round(time() - t0, 3), "seconds")

t1 = time()
pred = temp.predict(features_test)
print("predicting time:", round(time() - t1, 3), "seconds")
acc = accuracy_score(labels_test, pred)
print("accuracy by SVM method is : ", acc)

def TotalChrisPredicted(pred):
    count = 0
    for i in pred:
        if i == 1:
            count = count + 1
    return count

print("Total Chris predicted : ", TotalChrisPredicted(pred))
print (pred[10] , "\n" , pred[26], "\n", pred[50])

#########################################################


