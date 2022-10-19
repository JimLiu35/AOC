# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from control import lqr

''' Link 1 '''
S1 = np.array([0, 0]).reshape(2, 1)
p0 = np.array([0.5, 0.5]).reshape(2, 1)
A1 = p0 - S1

''' Link 2 '''
S2 = np.array([0, 0]).reshape(2, 1)

theta1 = np.linspace(-np.pi, np.pi, 300)
theta2 = np.linspace(-np.pi, np.pi, 300)
l1 = l2 = 1
r = 0.25
p0 = np.array([0.5, 0.5]).reshape(2, 1)
cspace = []

for i in theta1:
    D1 = np.array([l1 * np.cos(i), l1*np.sin(i)]).reshape(2, 1)
    D1norm = np.linalg.norm(D1)
    D1unit = D1 / D1norm
    f1 = S1 + max(min(A1.T @ D1unit, l1), 0) * D1unit
    distance1 = np.linalg.norm(f1 - p0)
    for k in theta2:
        S2 = D1
        pt = np.array([[np.cos(i) * l1 + np.cos(i + k) * l2],
                       [np.sin(i) * l1 + np.sin(i + k) * l2]])
        D2 = pt - S2
        A2 = p0 - S2
        D2norm = np.linalg.norm(D2)
        D2unit = D2 / D2norm
        f2 = S2 + max(min(A2.T @ D2unit, l2), 0) * D2unit
        distance2 = np.linalg.norm(f2 - p0)
        if min(distance1, distance2) >= r:
            cspace.append([i, k])

cspace = np.array([cspace])
m = cspace.shape[1]
cspace_plot = cspace.reshape(m, 2)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(cspace_plot[:, 0], cspace_plot[:, 1], c='g')
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')
ax.set_title('Configuration Space - No Points Penerate the Obstacle')
plt.show()
