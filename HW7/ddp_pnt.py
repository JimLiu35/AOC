import numpy as np
import matplotlib.pyplot as plt
from ddp_traj import ddp_traj
from ddp_cost import ddp_cost
from ddp import ddp


class Problem:

    def __init__(self):

        # time-step and # of segments
        self.h = 1
        self.N = 10

        # system mass
        self.m = 2

        # cost function specification
        self.Q = np.diag([0.01, 0.01, 0.005, 0.005])
        self.R = np.diag([0.1, 0.1])
        self.Pf = np.diag([1, 1, 5, 5])

        self.mu = 0

        # initial state
        self.x0 = np.array([-1, -1, 0.1, 0])

    def f(self, k, x, u):

        A = np.array([[1, 0, self.h, 0],
                      [0, 1, 0, self.h],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        B = np.array([[0, 0],
                      [0, 0],
                      [self.h, 0],
                      [0, self.h]])

        x = np.squeeze(A @ x + B @ u)

        return x, A, B

    def L(self, k, x, u):

        if k < self.N:
            L = self.h * 0.5 * (np.transpose(x)@self.Q @
                                x + np.transpose(u)@self.R@u)
            Lx = self.h * self.Q @ x
            Lxx = self.h * self.Q
            Lu = self.h * self.R @ u
            Luu = self.h * self.R
        else:
            L = np.transpose(x) @ self.Pf @ x * 0.5
            Lx = self.Pf @ x
            Lxx = self.Pf
            Lu = np.zeros(self.m)
            Luu = np.zeros((self.m, self.m))

        return L, Lx, Lxx, Lu, Luu


if __name__ == '__main__':

    prob = Problem()

    # initial control sequence
    us = np.concatenate((np.tile([[0.1], [0.05]], (1, prob.N//2)),
                         np.tile([[-0.1], [-0.05]], (1, prob.N//2))), axis=1)/2

    xs = ddp_traj(us, prob)
    V = ddp_cost(xs, us, prob)

    plt.figure()
    plt.plot(xs[0, :], xs[1, :], '-b')

    # two iterations should be enough for a linear-quadratic problem
    for i in range(2):
        dus, V, Vn, dV, a = ddp(us, prob)
        # update control
        us = us + dus
        prob.a = a
        xs = ddp_traj(us, prob)
        plt.plot(xs[0, :], xs[1, :], '-g')
    plt.plot(xs[0, :], xs[1, :], '-m')

    plt.show()
