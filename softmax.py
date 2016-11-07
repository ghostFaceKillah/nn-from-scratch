import numpy as np
import matplotlib.pyplot as plt
import pdb

ONES_CENTER = (2.0, 1.0)
ZEROS_CENTER = (-2.0, -1.0)

ONES_VAR = 4.0
ZEROS_VAR = 4.0

LEARNING_RATE = 0.01
LR = LEARNING_RATE

SAMPLE_SIZE = 100

def draw_sample(ones=True, sample_size=SAMPLE_SIZE):
    if ones:
        mean = ONES_CENTER
        var = ONES_VAR
    else:
        mean = ZEROS_CENTER
        var = ZEROS_VAR

    cov = np.array([
        [var, 0.0],
        [0.0, var]
    ])

    return np.random.multivariate_normal(mean, cov, sample_size)


def show_data_sample(ones, zeros, A, b):
    plt.scatter(ones[:, 0], ones[:, 1], c='b', marker='o')
    plt.scatter(zeros[:, 0], zeros[:, 1], c='r', marker='x')

    x = np.linspace(-2.0, 2.0, num=100)
    y = - A[0, 0] / A[0, 1] * x - b / A[0, 1]
    plt.plot(x, y)

    plt.xlim([-7, 7])
    plt.ylim([-7, 7])

    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_classifier_gradient(ones, zeros, A, b):
    from_x, to_x = -7.0, 7.0
    res = 100

    fake_x = np.array([[x, y]
                       for y in -np.linspace(from_x, to_x, res)
                       for x in np.linspace(from_x, to_x, res)])

    fake_y = sigmoid(np.dot(fake_x, A.T) + b)

    x = np.linspace(-2.0, 2.0, num=100)
    y = - A[0, 0] / A[0, 1] * x - b / A[0, 1]
    plt.plot(x, y)

    # That's our classification function
    img = fake_y.reshape((res, res))

    # make the actual plot
    plt.scatter(ones[:, 0], ones[:, 1], c='b', marker='o')
    plt.scatter(zeros[:, 0], zeros[:, 1], c='r', marker='x')

    plt.xlim([-7, 7])
    plt.ylim([-7, 7])

    plotlim = plt.xlim() + plt.ylim()
    plt.imshow(img, cmap=plt.cm.copper, interpolation='bicubic', alpha=1, extent=plotlim)
    plt.show()


def main():
    """  """
    A = np.random.normal(0.0, 0.1, (1, 2))
    b = np.random.normal(0.0, 0.1, 1)

    i = 0

    # for i in range(100):
    while True:
        ones = draw_sample(True)
        zeros = draw_sample(False)

        x = np.vstack([ones, zeros])
        y = np.append(np.ones(SAMPLE_SIZE), np.zeros(SAMPLE_SIZE))

        y_hat = sigmoid(np.dot(x, A.T) + b)

        err = y_hat[:, 0] - y

        dA = np.multiply(err[:, np.newaxis], x).mean(axis=0)
        db = err.mean()

        A -= LR * dA
        b -= LR * db

        # show_data_sample(ones, zeros, A, b)
        if i % 100 == 0:
            plot_classifier_gradient(ones, zeros, A, b)

        i += 1


if __name__ == '__main__':
    main()

