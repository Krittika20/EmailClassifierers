#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
import numpy as np
sys.path.append("ud120-projects/tools")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load(open("ud120-projects/final_project/final_project_dataset_unix.pkl", "rb"))

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below

print(data.argmax(axis=0))
max_value = data[67]

person = [key for key, feature in data_dict.items() if feature['salary'] == data[67][0]]
print(person)

data_dict.pop(person[0])

