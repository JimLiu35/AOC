# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from control import lqr


def stateEq(x10, x20, a, x2, u):
    x1 = np.zeros((1, 200), dtype='complex_')
    if u == 0:
        x1 = -1/a * (x2 - x20) + x10
    elif u == 1:
        x1 = -(1/a)**2 * (a * x2 + np.emath.log([1 - a*x2]))
    else:
        x1 = 1 / (a**2) * (-a * x2 + np.emath.log([1 + a*x2]))
    return np.real(x1)


# x1u0 = np.zeros((1, 200), dtype='complex_')
# x1u1 = np.zeros((1, 200), dtype='complex_')
# x1u_1 = np.zeros((1, 200), dtype='complex_')
x2 = np.linspace(2, 5, 200)
a = 10
x10 = 0 * np.ones_like(x2)
x20 = 0 * np.ones_like(x2)
x1u0 = stateEq(x10, x20, a, x2, 0)
x1u1 = stateEq(x10, x20, a, x2, 1)
x1u_1 = stateEq(x10, x20, a, x2, -1)

fig = plt.figure()
ax = fig.add_subplot(111)
# plt.plot(x2, x1u0)
plt.plot(x2.reshape(200, 1), x1u1)
plt.plot(x2.reshape(1, 200), x1u_1)

plt.show()
