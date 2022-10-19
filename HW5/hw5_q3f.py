# Imports
import numpy as np
import matplotlib.pyplot as plt

theta1 = np.linspace(0, 2*np.pi, 200)
theta2 = np.linspace(0, 2*np.pi, 200)
l1 = l2 = 1
r = 0.25
p0 = np.array([0.5, 0.5]).reshape(2, 1)
cspace = []
for i in theta1:
    for j in theta2:
        pt = np.array([[np.cos(i) * l1 + np.cos(i + j) * l2],
                       [np.sin(i) * l1 + np.sin(i + j) * l2]])
        distance = np.linalg.norm(pt - p0)
        if distance >= r:
            cspace.append([i, j])
cspace = np.array([cspace])
k = cspace.shape[1]
cspace_plot = cspace.reshape(k, 2)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(cspace_plot[:, 0], cspace_plot[:, 1], c='g')
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')
ax.set_title('Configuration Space - Tip Only')
plt.show()
