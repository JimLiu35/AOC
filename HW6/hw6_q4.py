# Imports
import numpy as np
import matplotlib.pyplot as plt

class HW6_Q4(object):
  """ Homework 6 Questions 4 """


  def __init__(self):

    # 4a) TODO: Implement me
    self.dt = 0.0
    self.A = np.zeros((2,2))
    self.B = np.zeros((2,1))
    self.w = np.zeros((2,1))
    self.Pf = np.array((2,2))
    self.Q = np.array((2,2))
    self.R = np.array((1,1))
    self.N = 0


  def calc_Ps_bs(self):
    """Calculate Ps and bs"""
    """Outputs: Ps (numpy.ndarray, shape: (2,2,101))
                bs (numpy.ndarray, shape: (2,1,101))"""

    # 4b) TODO: Implement me
    Ps = np.zeros((2, 2, self.N+1))
    bs = np.zeros((2, 1, self.N+1))
    return Ps, bs


  def control(self, x, P, b):
    """Calculate u_i given x_i, P_{i+1}, b_{i+1}"""
    """Inputs: x (numpy.ndarray, shape: (2))
               P (numpy.ndarray, shape: (2,2))
               b (numpy.ndarray, shape: (2,1))"""
    """Outputs: u (numpy.float64 or float or numpy.ndarray, shape: (1,1) or (1,))"""

    # 4c) TODO: Implement me
    u = 0.0
    return u


  def dynamics(self, x, u):
    """Calculate x_{i+1} given x_i, u_i"""
    """Inputs: x (numpy.ndarray, shape: (2,))
               u (numpy.float64 or float or numpy.ndarray, shape: (1,1) or (1,))"""
    """Outputs: x_ (numpy.ndarray, shape: (2,))"""

    # 4c) TODO: Implement me
    x_ = np.zeros(2)
    return x_


  def calc_xs_us(self, x, Ps, bs):
    """Calculate xs and us"""
    """Inputs: x (numpy.ndarray, size: 2)
               Ps (numpy.ndarray, size: 2x2x101)
               bs (numpy.ndarray, size: 2x1x101)"""
    """Outputs: xs (numpy.ndarray, size: 2x101)
                us (numpy.ndarray, size: 100)"""

    # 4c) TODO: Implement me
    xs = np.zeros((2, self.N+1))
    us = np.zeros((self.N))
    return xs, us


if __name__ == '__main__':
  """This code runs if you execute this script"""
  hw6_q4 = HW6_Q4()

  # # TODO: Uncomment the following lines to generate plots to visualize the 
  # # result of your functions
  # Ps, bs = hw6_q4.calc_Ps_bs()
  # x = np.array([10, 0])
  # xs, us = hw6_q4.calc_xs_us(x, Ps, bs)
  # plt.figure()
  # plt.plot(xs[0, :], xs[1, :])
  # plt.xlabel("Position")
  # plt.ylabel("Velocity")
  # plt.title("xI = [10, 0]")
  # x = np.array([10, 5])
  # xs, us = hw6_q4.calc_xs_us(x, Ps, bs)
  # plt.figure()
  # plt.plot(xs[0, :], xs[1, :])
  # plt.xlabel("Position")
  # plt.ylabel("Velocity")
  # plt.title("xI = [10, 5]")
  # x = np.array([10, -5])
  # xs, us = hw6_q4.calc_xs_us(x, Ps, bs)
  # plt.figure()
  # plt.plot(xs[0, :], xs[1, :])
  # plt.xlabel("Position")
  # plt.ylabel("Velocity")
  # plt.title("xI = [10, -5]")
  # plt.show()