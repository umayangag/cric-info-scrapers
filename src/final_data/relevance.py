import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def relevance(x):
    x = np.array(x)
    return sigmoid(x - 50)
