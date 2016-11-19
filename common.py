import numpy as np

def binarize(pre_y):
    """ 
    binarize output from class number to indicator array
    e.g. [2, 1] -> [[0, 0, 1, 0], [0, 1, 0, 0]] 
    """
    y = np.zeros((pre_y.shape[0], 1 + pre_y.max()))
    y[range(pre_y.shape[0]), pre_y.astype(int)] = 1.0
    return y


def unbinarize(y):
    """ [[0, 0, 1, 0], [0, 1, 0, 0]] -> [2, 1] """
    return np.sum(y * np.array(range(0,  y.shape[1])), axis=1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    x[x < 0] = 0
    return x


def draw_params(size):
    return np.random.normal(0.0, 0.1, size)


def elu(x, alpha=1.0):
    y = x.copy()
    y[x < 0] = alpha * (np.exp(y[x < 0]) - 1)
    return y

def d_elu(x, alpha=1.0):
    y = elu(x)
    y[y > 0] = 1.0
    y[y <= 0] += alpha
    return y
