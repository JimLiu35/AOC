import numpy as np
import matplotlib.pyplot as plt

# Kalman filtering of the double integrator with position


class Problem:

  def __init__(self):

    # timing
    self.dt = 0.1   # time-step
    self.N = 100    # total time-steps
    self.T = self.N * self.dt   # final time

    # noise terms
    self.q = .3    # external disturbance variance
    self.r = .9    # measurement noise variance

    # F matrix
    self.F = np.array([[1, self.dt], [0, 1]])

    # G matrix
    self.G = np.array([[self.dt**2 / 2],
                       [self.dt]])

    # Q matrix
    self.Q = self.q * np.array([[self.dt**3 / 3, self.dt**2 / 2],
                                [self.dt**2 / 2, self.dt]])

    # R matrix
    self.R = self.r

    # H matrix
    self.H = np.array([[1, 0]])


def kf_predict(x, P, u, prob):

  x = prob.F @ x + prob.G @ u
  P = prob.F @ P @ np.transpose(prob.F) + prob.Q
  return x, P


def kf_correct(x, P, z, prob):

  K = P @ np.transpose(prob.H) @ np.linalg.inv(prob.H @ P @ np.transpose(prob.H) +
                                               prob.R)
  P = (np.eye(np.size(x)) - K @ prob.H) @ P
  x = x + K @ (z - prob.H @ x)
  return x, P


prob = Problem()

# initial estimate of mean and covariance
x = np.array([0, 0])
P = 10 * np.diag([1, 1])

xts = np.zeros((2, prob.N + 1))     # true states
xs = np.zeros((2, prob.N + 1))      # estimated states
Ps = np.zeros((2, 2, prob.N + 1))   # estimated covariances

zs = np.zeros((1, prob.N))        # estimated state

pms = np.zeros((1, prob.N))       # measured position

xts[:, 0] = x
xs[:, 0] = x
Ps[:, :, 0] = P

for k in range(prob.N):
  u = np.array([np.cos(k / prob.N)])  # pick some known control

  # true state
  xts[:, k + 1] = prob.F @ xts[:, k] + \
      prob.G @ (u + np.sqrt(prob.q) * np.random.randn(1))

  x, P = kf_predict(x, P, u, prob)  # prediction

  # generate random measurement
  z = xts[0, k + 1] + np.sqrt(prob.r) * np.random.randn(1)

  x, P = kf_correct(x, P, z, prob)  # correction

  # record result
  xs[:, k + 1] = x
  Ps[:, :, k + 1] = P
  zs[:, k] = z

plt.plot(xts[0, :], '--', linewidth=2)
plt.plot(xs[0, :], 'g', linewidth=2)
plt.plot(range(1, prob.N + 1), zs[0, :], 'r', linewidth=2)

plt.legend({'true', 'estimated', 'measured'})

# 95% confidence intervals of the estimated position
plt.plot(xs[0, :] + 1.96 * np.transpose(np.reshape(np.sqrt(Ps[0, 0, :]),
                                                   prob.N + 1)), '-g')
plt.plot(xs[0, :] - 1.96 * np.transpose(np.reshape(np.sqrt(Ps[0, 0, :]),
                                                   prob.N + 1)), '-g')

plt.show()
