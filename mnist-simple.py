"""
# MNIST NN - simple architecture

The simple architecure

"""
from common import binarize, unbinarize, sigmoid, relu, draw_params, elu, d_elu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data
import tqdm

IMG_W = IMG_H = 28
IMG_SIZE = IMG_W * IMG_H
NUMBER_OF_TRAINING_ITERATIONS = 20000
BATCH_SIZE = 64
HIDDEN_1_SIZE = 40
HIDDEN_2_SIZE = 40
OUTPUT_CLASSES_NO = 10
LR = 0.01


def get_data_provider():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return mnist


def draw_params(size):
    return np.random.normal(0.0, 0.1, size)


def train_model():
    # Step 0 - pull data provider 
    mnist = get_data_provider()

    # Step 1 - build the model
    W_1 = draw_params((HIDDEN_1_SIZE, IMG_SIZE))
    b_1 = draw_params((HIDDEN_1_SIZE, 1))

    W_2 = draw_params((HIDDEN_2_SIZE, HIDDEN_1_SIZE))
    b_2 = draw_params((HIDDEN_2_SIZE, 1))

    W_3 = draw_params((OUTPUT_CLASSES_NO, HIDDEN_2_SIZE))
    b_3 = draw_params((OUTPUT_CLASSES_NO, 1))

    # Step 3 - train the model
    accuracy_acc = []
    for iter_no in tqdm.tqdm(xrange(NUMBER_OF_TRAINING_ITERATIONS)):
        x, y = mnist.train.next_batch(BATCH_SIZE)

        # forward pass
        a_1 = np.dot(x, W_1.T) + b_1.T
        h_1 = elu(a_1)
        a_2 = np.dot(h_1, W_2.T) + b_2.T
        h_2 = elu(a_2)
        y_hat = sigmoid(np.dot(h_2, W_3.T) + b_3.T)

        err = y_hat - y

        # some accounting
        pred_class = np.argmax(y_hat, axis=1)
        accuracy = accuracy_score(unbinarize(y), pred_class)
        accuracy_acc.append(accuracy)

        # backward pass
        dW_3 = np.dot(err.T, h_2) / h_2.shape[0]
        db_3 = err.mean()

        dh_2 = np.dot(err, W_3)
        da_2 = dh_2 * d_elu(a_2)

        dW_2 = np.dot(da_2.T, h_1) / h_1.shape[0]
        db_2 = da_2.mean()

        dh_1 = np.dot(da_2, W_2)
        da_1 = dh_1 * d_elu(a_1)

        dW_1 = np.dot(da_1.T, x) / x.shape[0]
        db_1 = da_1.mean()

        # apply parameter learning
        W_1 -= LR * dW_1
        b_1 -= LR * db_1
        
        W_2 -= LR * dW_2
        b_2 -= LR * db_2

        W_3 -= LR * dW_3
        b_3 -= LR * db_3

    return accuracy_acc


if __name__ == '__main__':
    pd.Series(train_model()).plot()
    plt.show()
