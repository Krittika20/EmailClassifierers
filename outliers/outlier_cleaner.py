#!/usr/bin/python
import numpy as np
from numpy import array_equal

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    max_value = 0
    # ### your code goes here
    residual_error = list(abs(np.array(predictions) - np.array(net_worths)))

    ages = list(np.array(ages))
    net_worths = list(np.array(net_worths))

    clean_length = 0.1 * (len(residual_error))
    max_value = max(residual_error)
    length = 0

    while (length <= clean_length):
        for i in range(int(0.1 * len(residual_error))):
            idx = residual_error.index(max_value)
            del residual_error[idx]
            del ages[idx]
            del net_worths[idx]
            max_value = max(residual_error)
            length = length + 1

    for i in range(len(ages)):
        cleaned_data.insert(i, (ages[i], net_worths[i], residual_error[i]))

    return cleaned_data

