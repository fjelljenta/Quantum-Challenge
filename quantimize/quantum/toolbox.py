from quantimize.data_access import get_flight_info
from quantimize.converter import datetime_to_seconds
import numpy as np
from sklearn import preprocessing


def tensorize_flight_info():
    """
    Transforms the flight info into a tensor of shape (M, 6), with M being the number of flights in the data
    :param flight_number_list: a list of indices of flights
    :return: the tensorized flight data
    """
    res = []
    for i in [i for i in range(41)] + [i for i in range(42, 100)]:
        line = list(get_flight_info(i).values())
        res.append([datetime_to_seconds(line[0])] + line[1:])
    return np.array(res)


def normalize_input_data(data):
    """
    normalize data in tensor form with respect to individual features
    :param data: data in tensor form
    :return: normalized data in tensor form
    """
    scaler = preprocessing.StandardScaler().fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled


def sigmoid(x, c):
    return 1 / (1+np.exp(-c*x))
