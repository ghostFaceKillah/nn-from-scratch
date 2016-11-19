import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import tqdm

from common import binarize, unbinarize, sigmoid, relu, draw_params, elu, d_elu


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


def show_elu_properties():
    x = np.linspace(-5., 5., 100)
    pd.DataFrame({
        'f': elu(x),
        'df': d_elu(x)
    }).plot()
    plt.show()


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


def one_relu_run():
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

    # show_data_sample(x, y)
    return accuracy_acc


def one_elu_run():
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
        y_1 = elu(a)
        y_hat = sigmoid(np.dot(y_1, W_2.T) + b_2.T)

        err = y_hat - y

        pred_class = np.argmax(y_hat, axis=1)
        accuracy = accuracy_score(unbinarize(y), pred_class)
        accuracy_acc.append(accuracy)

        # backward pass
        dW_2 = np.dot(err.T, y_1) / y_1.shape[0]
        db_2 = err.mean()

        dy_1 = np.dot(err, W_2)
        da = dy_1 * d_elu(a)
        
        dW_1 = np.dot(da.T, x) / x.shape[0]
        db_1 = da.mean()

        W_1 -= LR * dW_1
        b_1 -= LR * db_1

        W_2 -= LR * dW_2
        b_2 -= LR * db_2

    # show_data_sample(x, y)
    return accuracy_acc


if __name__ == '__main__':
    # accuracy_acc = one_relu_run()
    pd.DataFrame({
        'elu': one_elu_run(), 
        'relu': one_relu_run()
    }).plot()
    plt.show()
