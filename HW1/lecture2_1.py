import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

f1 = lambda x: np.transpose(x)@np.array([[1, -1], [-1, 4]])@x
f2 = lambda x: np.transpose(x)@np.array([[-1, 1], [1, 3]])@x
f3 = lambda x: (x[0]-x[1]**2)*(x[0]-2*x[1]**2)

x1 = np.arange(-5, 5, 0.01)
x2 = np.arange(-5, 5, 0.01)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros((X1.shape[0], X1.shape[1]))
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i, j] = f1(np.array([X1[i, j], X2[i, j]]))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X1, X2, Z, cmap=cm.jet, linewidth=0,
                          antialiased=False)

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i, j] = f2(np.array([X1[i, j], X2[i, j]]))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X1, X2, Z, cmap=cm.jet, linewidth=0,
                          antialiased=False)

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        Z[i, j] = f3(np.array([X1[i, j], X2[i, j]]))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X1, X2, Z, cmap=cm.jet, linewidth=0,
                          antialiased=False)

plt.show()
