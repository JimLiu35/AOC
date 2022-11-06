import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Example of trajectory optimization for a car-like model using shooting and
# least-squares Gauss Newton method

class Problem:

  def __init__(self):

    # time horizon and segments
    self.tf = 5
    self.N = 16
    self.h = self.tf / self.N
    self.m = 2
    self.n = 5

    # cost function parameters
    self.Q = np.diag([0, 0, 0, 0, 0])
    self.R = np.diag([0.1, 0.5])
    self.Qf = np.diag([10, 10, 10, 1, 1])
    self.Qs = np.sqrt(self.Q)
    self.Rs = np.sqrt(self.R)
    self.Qfs = np.sqrt(self.Qf)

    # initial state
    self.x0 = np.array([3, -2, -2.2, 0, 0])

  def f(self, k, x, u):
    # car dynamics and jacobians

    h = self.h
    c = np.cos(x[2])
    s = np.sin(x[2])
    v = x[3]
    w = x[4]

    A = np.array([[1, 0, -h*s*v, h*c, 0],
                  [0, 1, h*c*v, h*s, 0],
                  [0, 0, 1, 0, h],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])

    B = np.array([[0, 0],
                  [0, 0],
                  [0, 0],
                  [h, 0],
                  [0, h]])

    x = np.array([x[0] + h*c*v, x[1] + h*s*v, x[2] + h*w, v + h*u[0], 
      w + h*u[1]])

    return x, A, B    


  def sys_traj(self, us):

    N = us.shape[1]

    xs = np.zeros((self.n, N+1))
    xs[:, 0] = self.x0
    for k in range(N):
      xs[:, k+1], _, _ = self.f(k, xs[:, k], us[:, k])

    return xs

  def car_cost(self, us):
    # the car costs in least-squares form, i.e. the residuals at each time-step

    us = us.reshape((self.m, self.N))
    xs = self.sys_traj(us)

    y = np.zeros(self.N*(self.n+self.m))
    for k in range(N):
      y[k*self.m:(k+1)*self.m] = self.Rs@us[:, k]
    for k in range(0, N-1):
      y[self.N*self.m+k*self.n:self.N*self.m+(k+1)*self.n] = self.Qs@xs[:, k+1]
    y[self.N*self.m+(self.N-1)*self.n:] = self.Qfs@xs[:, N]
    
    return y


if __name__ == '__main__':

  prob = Problem()

  # initial control sequence
  us = np.zeros((2, prob.N))
  xs = prob.sys_traj(us)

  m = prob.m
  N = prob.N
  lb = np.concatenate((np.ones(prob.N)*-1.0, np.ones(prob.N)*-0.4))
  ub = np.concatenate((np.ones(prob.N)*1.0, np.ones(prob.N)*0.4))

  res = least_squares(prob.car_cost, us.flatten(), bounds=(lb, ub))
  us = np.reshape(res.x, (prob.m, prob.N))
  xs = prob.sys_traj(us)
  
  fig, axs = plt.subplots(1, 2)
  axs[0].plot(xs[0,:], xs[1,:], '-b')
  axs[0].set_xlabel('x')
  axs[0].set_ylabel('y')
  axs[1].plot(np.arange(0, prob.tf, prob.h), us[0, :])
  axs[1].plot(np.arange(0, prob.tf, prob.h), us[1, :])
  axs[1].set_xlabel('sec.')
  axs[1].legend(["u_0", "u_1"])
  plt.subplots_adjust(wspace=0.25)

  y = prob.car_cost(us)
  J = np.transpose(y)@y / 2.0
  print("cost = ", J)

  plt.show()
