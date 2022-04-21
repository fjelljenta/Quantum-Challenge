from quantimize.data_access import get_flight_info
from quantimize.converter import datetime_to_seconds
import numpy as np
from sklearn import preprocessing


def tensorize_flight_info():
    """Transforms the flight info into a tensor of shape (M, 6), with M being the number of flights in the data

    Args:
        flight_number_list: a list of indices of flights numbers

    Returns:
        the tensorized flight data
    """
    res = []
    for i in [i for i in range(41)] + [i for i in range(42, 100)]:
        line = list(get_flight_info(i).values())
        res.append([datetime_to_seconds(line[0])] + line[1:])
    return np.array(res)


def normalize_input_data(data):
    """normalize data in tensor form with respect to individual features

    Args:
        data: data in tensor form

    Returns:
        normalized data in tensor form
    """
    scaler = preprocessing.StandardScaler().fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled


def sigmoid(x, c):
    """Returns the sigmoid function value of x and c

    Args:
        x: parameter of the function
        c: parameter of the function

    Returns:
        sigmoid function value

    """
    return 1 / (1+np.exp(-c*x))
