from matplotlib.pyplot import figure, show, cm
import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(0., 1., 100)
ys = np.linspace(0., 1., 100)

x = np.linspace(0., 1.0, 100)
y = x ** 2

X = np.dot(xs[:, np.newaxis], ys[:, np.newaxis].T)

plt.plot(x, y)

plotlim = plt.xlim() + plt.ylim()  
plt.imshow(X, cmap=cm.copper, interpolation='bicubic', alpha=1, extent=plotlim)
plt.show()

