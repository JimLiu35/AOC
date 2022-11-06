import numpy as np
import matplotlib.pyplot as plt
from ddp_traj import ddp_traj
from ddp_cost import ddp_cost
from ddp import ddp


class Problem:

    def __init__(self):

        # time horizon and segments
        self.tf = 10.0
        self.N = 32
        self.h = self.tf / self.N

        # cost function parameters
        self.Q = np.diag([0, 0, 0, 0, 0])
        self.R = np.diag([1, 5])
        self.Pf = np.diag([5, 5, 1, 1, 1])

        # initial state
        self.x0 = np.array([-5, -2, -1.2, 0, 0])

        self.mu = 1

        self.os_p = [[-2.5, -2.5]]
        self.os_r = [1]
        self.ko = 1

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
            Lu = np.zeros(2)
            Luu = np.zeros((2, 2))

        if hasattr(self, 'os_r'):
            for i in range(len(self.os_r)):
                g = x[:2] - self.os_p[i]
                c = self.os_r[i] - np.linalg.norm(g)
                if c < 0:
                    continue

                L = L + self.ko/2.0*c**2
                v = g/np.linalg.norm(g)
                Lx[:2] = Lx[:2] - self.ko*c*v
                Lxx[:2, :2] = Lxx[:2, :2] + self.ko*np.transpose(v)@v

        return L, Lx, Lxx, Lu, Luu


if __name__ == '__main__':

    prob = Problem()

    # initial control sequence
    us = np.concatenate((np.tile([[0.1], [0.1]], (1, prob.N//2)),
                         np.tile([[-0.1], [-0.1]], (1, prob.N//2))), axis=1)

    xs = ddp_traj(us, prob)
    Jmin = ddp_cost(xs, us, prob)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(xs[0, :], xs[1, :], '-b')
    for j in range(len(prob.os_r)):
        circle = plt.Circle(prob.os_p[j], prob.os_r[j], color='r', linewidth=2,
                            fill=False)
        axs[0].add_patch(circle)
    axs[0].axis('equal')
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    for i in range(50):
        dus, V, Vn, dV, a = ddp(us, prob)
        # update control
        us = us + dus
        prob.a = a
        prob.ko = prob.ko*1.1
        xs = ddp_traj(us, prob)
        axs[0].plot(xs[0, :], xs[1, :], '-b')
    axs[0].plot(xs[0, :], xs[1, :], '-g', linewidth=3)

    axs[1].plot(np.arange(0, prob.tf, prob.h), us[0, :])
    axs[1].plot(np.arange(0, prob.tf, prob.h), us[1, :])
    axs[1].set_xlabel("sec.")
    axs[1].legend(["u_1", "u_2"])

    plt.show()
