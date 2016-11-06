import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import tqdm

CLASSES = {
    0: ( 2.0,  2.0),
    1: ( 2.0, -2.0),
    2: (-2.0, -2.0),
    3: (-2.0,  2.0)
}

VAR = 1.0

LEARNING_RATE = 0.01
LR = LEARNING_RATE

SAMPLE_SIZE = 100

def draw_sample(sample_size=SAMPLE_SIZE):
    def draw(mean, covar, ssize):
        return np.random.multivariate_normal(mean, covar, sample_size)
    
    cov = np.array([
        [VAR, 0.0],
        [0.0, VAR]
    ])


    x = np.vstack([
        draw(mean, cov, sample_size) for _, mean in CLASSES.iteritems()
    ])

    pre_y = np.concatenate([
        classname * np.ones(sample_size, dtype=np.int)
        for classname, _ in CLASSES.iteritems()
    ])

    y = binarize(pre_y)

    return x, y


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


def show_data_sample(x, y, A=None, b=None):
    """ Plot a sample of data """
    class_marks = zip(range(4), ['o', 'x', 'v', '$\heartsuit$'], 'bryg')
    y = unbinarize(y)

    for classname, marker, color in class_marks:
        data = x[y == classname]
        plt.scatter(data[:, 0], data[:, 1], c=color, marker=marker, s=30)

    if A is not None and b is not None:
        x = np.linspace(-2.0, 2.0, num=100)
        y = - A[0, 0] / A[0, 1] * x - b / A[0, 1]
        plt.plot(x, y)

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])

    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    x[x < 0] = 0
    return x

def draw_params(size):
    return np.random.normal(0.0, 0.1, size)

W_1 = draw_params((4, 2))
b_1 = draw_params((4, 1))
W_2 = draw_params((4, 4))
b_2 = draw_params((4, 1))

i = 0

accuracy_acc = []

for i in tqdm.tqdm(range(7000)):
    x, y = draw_sample()

    # forward pass
    a = np.dot(x, W_1.T) + b_1.T
    y_1 = relu(a)
    y_hat = sigmoid(np.dot(y_1, W_2.T) + b_2.T)

    err = y_hat - y

    pred_class = np.argmax(y_hat, axis=1)
    accuracy = accuracy_score(unbinarize(y), pred_class)
    accuracy_acc.append(accuracy)

    # backward pass
    dW_2 = np.dot(err.T, y_1) / y_1.shape[0]
    db_2 = err.mean()

    dy_1 = np.dot(err, W_2)
    da = dy_1 * (a > 0).astype(float)
    
    dW_1 = np.dot(da.T, x) / x.shape[0]
    db_1 = da.mean()

    W_1 -= LR * dW_1
    b_1 -= LR * db_1

    W_2 -= LR * dW_2
    b_2 -= LR * db_2

show_data_sample(x, y)

pd.Series(accuracy_acc).plot()
plt.show()
