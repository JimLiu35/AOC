import numpy as np
import matplotlib.pyplot as plt
from ddp_traj import ddp_traj
from ddp_cost import ddp_cost
from ddp import ddp


class Problem:

    def __init__(self):

        # time-step and # of segments
        self.tf = 10
        self.N = 100
        self.h = self.tf / self.N

        # system mass
        self.m = 2

        # cost function specification
        self.Q = np.diag([0.01, 0.01, 0.005, 0.005])
        self.R = np.diag([0.1, 0.1])
        self.Pf = np.diag([1, 1, 5, 5])

        self.mu = 1

        self.os_p = [[-2.5, -2.5]]
        self.os_r = [1]
        # upper constraint for u
        self.u_con = np.array([0.15, 0.15])
        # lower constraint for u
        self.l_con = np.array([-0.15, 0.15])
        self.ko = 10

        # initial state
        self.x0 = np.array([-5, -5, 0.3, 0])

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

        if hasattr(self, 'u_con'):
            if k < self.N:
                c = np.zeros_like(g)
                for i in range(self.u_con.shape[0]):
                    c_temp = u[i] - self.u_con[i]
                    if c_temp < 0:
                        c[i] = 0
                    else:
                        c[i] = c_temp
                L = L + self.ko/2.0*c[0]**2 + self.ko/2.0*c[1]**2
                Lu = Lu + self.ko*c
                Luu = Luu + self.ko * np.eye(2)

        if hasattr(self, 'l_con'):
            if k < self.N:
                c = np.zeros_like(g)
                # print(self.u_con.shape[0])
                for i in range(self.l_con.shape[0]):
                    c_temp = - u[i] + self.l_con[i]
                    if c_temp < 0:
                        c[i] = 0
                    else:
                        c[i] = c_temp
                L = L + self.ko/2.0*c[0]**2 + self.ko/2.0*c[1]**2
                Lu = Lu - self.ko*c
                Luu = Luu + self.ko * np.eye(2)

        return L, Lx, Lxx, Lu, Luu


if __name__ == '__main__':

    prob = Problem()

    # initial control sequence
    us = np.concatenate((np.tile([[0.1], [0.05]], (1, prob.N//2)),
                         np.tile([[-0.1], [-0.05]], (1, prob.N//2))), axis=1)/2

    xs = ddp_traj(us, prob)
    V = ddp_cost(xs, us, prob)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(xs[0, :], xs[1, :], '-b')
    for j in range(len(prob.os_r)):
        circle = plt.Circle(prob.os_p[j], prob.os_r[j], color='r', linewidth=2,
                            fill=False)
        axs[0].add_patch(circle)
    axs[0].axis('equal')

    for i in range(20):
        dus, V, Vn, dV, a = ddp(us, prob)
        # update control
        us = us + dus
        prob.a = a
        xs = ddp_traj(us, prob)
        axs[0].plot(xs[0, :], xs[1, :], '-g')
        prob.ko = prob.ko*1.1
    axs[0].plot(xs[0, :], xs[1, :], '-m')

    axs[1].plot(np.arange(0, prob.tf, prob.h), us[0, :])
    axs[1].plot(np.arange(0, prob.tf, prob.h), us[1, :])
    axs[1].set_xlabel("sec.")
    axs[1].legend(["u_1", "u_2"])

    plt.show()
