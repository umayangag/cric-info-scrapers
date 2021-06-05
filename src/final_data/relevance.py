import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def relevance(x,offset):
    x = np.array(x)
    y = np.array([])
    for value in x:
        y = np.append(y, [sigmoid(value-offset)])
    return y
