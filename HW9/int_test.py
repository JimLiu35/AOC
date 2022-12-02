import numpy as np
import matplotlib.pyplot as plt

# Kalman filtering of the double integrator with position measurements


class Problem:

    def __init__(self):

        # timing
        self.dt = 1   # time-step
        self.N = 30    # total time-steps
        self.T = self.N * self.dt   # final time

        # noise terms
        self.qu = (3e-9)**2  # external disturbance variance thetadot
        self.qv = (3e-6)**2  # external disturbance variance bias
        self.qn = (1.5e-5)**2  # measurement noise variance

        # PHI matrix
        self.Phi = np.array([[1, -self.dt], [0, 1]])

        # G matrix
        self.G = np.array([[self.dt], [0]])

        # Q matrix
        self.Q = np.array([[self.qv * self.dt + self.qu * (self.dt**3 / 3.0),
                            -self.qu * (self.dt**2 / 2.0)],
                           [-self.qu * (self.dt**2 / 2.0), self.qu * self.dt]])

        # R matrix
        self.R = self.qn

        # H matrix
        self.H = np.array([[1, 0]])


def kf_predict(x, P, u, prob):

    x = prob.Phi @ x + prob.G @ u
    P = prob.Phi @ P @ np.transpose(prob.Phi) + prob.Q
    return x, P


def kf_correct(x, P, z, prob):

    K = P @ np.transpose(prob.H) @ np.linalg.inv(prob.H @ P @ np.transpose(prob.H) +
                                                 prob.R)
    P = (np.eye(np.size(x)) - K @ prob.H) @ P
    x = x + K @ (z - prob.H @ x)
    return x, P


prob = Problem()

# initial estimate of mean and covariance
x = np.array([0, 1.7e-7])
P = np.diag([1e-2, 1e-12])
thetadot = 0.02  # given trajectory for true state theta

xts = np.zeros((2, prob.N + 1))     # true states
xs = np.zeros((2, prob.N + 1))      # estimated states
Ps = np.zeros((2, 2, prob.N + 1))   # estimated covariances
nm = np.zeros((prob.N + 1, 1))      # Norm of covariance

zs = np.zeros((1, prob.N))        # estimated state

pms = np.zeros((1, prob.N))       # measured position

xts[:, 0] = x
xs[:, 0] = x
Ps[:, :, 0] = P
nm[0] = np.linalg.norm(P)

for k in range(prob.N):

    xts[:, k + 1] = xts[:, k] + np.concatenate((np.array([thetadot * prob.dt]),
                                                np.sqrt(prob.qu) * np.random.randn(1)))

    # generate u based on true state
    u = thetadot + xts[1, k + 1] + np.sqrt(prob.qv) * np.random.randn(1)

    x, P = kf_predict(x, P, u, prob)  # prediction

    # generate random measurement
    z = xts[0, k + 1] + np.sqrt(prob.qn) * np.random.randn(1)

    x, P = kf_correct(x, P, z, prob)  # correction

    # record result
    xs[:, k + 1] = x
    Ps[:, :, k + 1] = P
    zs[:, k] = z + 1
    nm[k + 1] = np.linalg.norm(P)

plt.figure()
plt.plot(xts[0, :], 'x--', linewidth=2)
plt.plot(xs[0, :], 'gx-', linewidth=2)
plt.plot(prob.dt * np.arange(1, prob.N + 1), zs[0, :], 'ro-', linewidth=2)

plt.xlabel('time (sec)')
plt.ylabel('theta (rad)')
plt.legend({'true', 'estimated', 'measured'})

# 95% confidence intervals of the estimated position
plt.plot(xs[0, :] + 1.96 * np.transpose(np.reshape(np.sqrt(Ps[0, 0, :]),
                                                   prob.N + 1)), '-g')
plt.plot(xs[0, :] - 1.96 * np.transpose(np.reshape(np.sqrt(Ps[0, 0, :]),
                                                   prob.N + 1)), '-g')

plt.figure()
plt.plot(prob.dt * np.arange(1, prob.N + 1), nm[1:], '-')
plt.legend("norm covariance")
plt.xlabel("time (sec)")

plt.figure()
error = xs - xts
plt.plot(prob.dt * np.arange(prob.N + 1), error[0, :], 'r')
plt.plot(prob.dt * np.arange(prob.N + 1), error[1, :], 'b')
plt.ylabel("rad or rad/s")
plt.xlabel("time (s)")
plt.legend({"etheta", "ebias"})

plt.show()
