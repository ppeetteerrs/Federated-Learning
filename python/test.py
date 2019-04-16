import pickle
import os
import numpy as np
from collections import OrderedDict
from random import sample, choice
import tensorflow as tf
import json


def load_h5():
    with open("cifar/test_data.h5", "rb") as data_file:
        X, Y = pickle.load(data_file)
    Y = np.asarray([[item] for item in Y])
    with open("cifar/test_data.h5", "wb") as data_file:
        pickle.dump([X, Y], data_file)


load_h5()
