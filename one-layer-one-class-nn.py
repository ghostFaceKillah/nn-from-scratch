import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LD_ONES_CENTER = (-2.0, -2.0)
RU_ONES_CENTER = (2.0, 2.0)

LU_ZEROS_CENTER = (-2.0, 2.0)
RD_ZEROS_CENTER = (2.0, -2.0)

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
        draw(LD_ONES_CENTER, cov, sample_size),
        draw(RU_ONES_CENTER, cov, sample_size),
        draw(LU_ZEROS_CENTER, cov, sample_size),
        draw(RD_ZEROS_CENTER, cov, sample_size)
    ])
    y = np.append(np.ones(2 * sample_size), np.zeros(2 * sample_size))

    return x, y


def show_data_sample(x, y, A=None, b=None):
    ones = x[y == 1]
    zeros = x[y == 0]
    plt.scatter(ones[:, 0], ones[:, 1], c='b', marker='o')
    plt.scatter(zeros[:, 0], zeros[:, 1], c='r', marker='x')

    if A is not None and b is not None:
        x = np.linspace(-2.0, 2.0, num=100)
        y = - A[0, 0] / A[0, 1] * x - b / A[0, 1]
        plt.plot(x, y)

    plt.xlim([-7, 7])
    plt.ylim([-7, 7])

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
W_2 = draw_params((1, 4))
b_2 = draw_params((1, 1))

i = 0

err_acc = []

for i in range(5000):
    x, y = draw_sample()

    # forward pass
    a = np.dot(x, W_1.T) + b_1.T
    y_1 = relu(a)
    y_hat = sigmoid(np.dot(y_1, W_2.T) + b_2.T)

    err = y_hat[:, 0] - y

    pred = (y_hat > 0.5).astype(float)[:, 0]
    err_rate = ((y - pred)** 2).mean()
    err_acc.append(err_rate)

    # backward pass 
    dW_2 = np.dot(err[:, np.newaxis].T, y_1) / y_1.shape[0]
    db_2 = err.mean()

    dy_1 = np.dot(err[:, np.newaxis], W_2)
    da = dy_1 * (a > 0).astype(float)
    
    dW_1 = np.dot(da.T, x) / x.shape[0]
    db_1 = da.mean()

    W_1 -= LR * dW_1
    b_1 -= LR * db_1

    W_2 -= LR * dW_2
    b_2 -= LR * db_2

show_data_sample(x, y)

pd.Series(err_acc).plot()
plt.show()
