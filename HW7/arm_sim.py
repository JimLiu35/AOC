import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Example simulation of two-link arm dynamics using discrete dynamics


class Arm:

    def __init__(self):

        # model parameters
        self.m1 = 1                     # mass of first body
        self.m2 = 1                     # mass of second body
        self.l1 = 0.5                   # length of first body
        self.l2 = 0.5                   # length of second body
        self.lc1 = 0.25                 # distance to COM
        self.lc2 = 0.25                 # distance to COM
        self.I1 = self.m1*self.l1/12.0  # inertia 1
        self.I2 = self.m2*self.l2/12.0  # inertia 2
        self.g = 9.8                    # gravity
        self.tf = 2.0                   # final time
        self.N = 128                    # number of time steps
        self.h = self.tf / self.N       # time-step
        self.n = 4                      # Dimension of state
        self.m = 2
        self.x0 = np.zeros(4)

        # cost function parameters
        # self.Q = np.diag([0, 0, 0, 0])
        self.Q = np.diag([1, 1, 1, 1])
        self.R = np.diag([0.1, 0.5])
        self.Qf = np.diag([1, 1, 1, 1])
        self.Qs = np.sqrt(self.Q)
        self.Rs = np.sqrt(self.R)
        self.Qfs = np.sqrt(self.Qf)

    def f(self, k, x, u):
        # arm discrete dynamics
        # set jacobians A, B to [] if unavailable

        q = x[:2]
        v = x[2:4]

        c1 = np.cos(q[0])
        c2 = np.cos(q[1])
        s2 = np.sin(q[1])
        c12 = np.cos(q[0]+q[1])

        # coriolis matrix
        C = -self.m2*self.l1*self.lc2*s2*np.array([[v[1], v[0]+v[1]], [-v[0], 0]]) \
            + np.diag([0.2, 0.2])

        # mass elements
        m11 = self.m1*self.lc1**2 + self.m2*(self.l1**2 + self.lc2**2 +
                                             2*self.l1*self.lc2*c2) + self.I1 + self.I2
        m12 = self.m2*(self.lc2**2 + self.l1*self.lc2*c2) + self.I2
        m22 = self.m2*self.lc2**2 + self.I2

        # mass matrix
        M = np.array([[m11, m12], [m12, m22]])

        # gravity vector
        fg = np.array([(self.m1*self.lc1 + self.m2*self.l1)*self.g*c1 +
                       self.m2*self.lc2*self.g*c12, self.m2*self.lc2*self.g*c12])

        # acceleration
        a = np.linalg.inv(M) @ (u - C@v - fg)
        v = v + self.h*a

        x = np.concatenate((q + self.h*v, v))

        # leave empty to use finite difference approximation
        A = np.array([])
        B = np.array([])

        return x, A, B

    def sys_traj(self, us):

        N = us.shape[1]

        xs = np.zeros((self.n, N+1))
        xs[:, 0] = self.x0
        for k in range(N):
            xs[:, k+1], _, _ = self.f(k, xs[:, k], us[:, k])

        return xs

    def arm_cost(self, us):
        # the car costs in least-squares form, i.e. the residuals at each time-step

        us = us.reshape((self.m, self.N))
        xs = self.sys_traj(us)

        y = np.zeros(self.N*(self.n+self.m))
        N = self.N
        for k in range(N):
            y[k*self.m:(k+1)*self.m] = self.Rs@us[:, k]
        for k in range(0, N-1):
            y[self.N*self.m+k*self.n:self.N*self.m +
                (k+1)*self.n] = self.Qs@xs[:, k+1]
        y[self.N*self.m+(self.N-1)*self.n:] = self.Qfs@xs[:, N]

        return y


if __name__ == '__main__':

    arm = Arm()

    # initial state
    x0 = np.zeros(4)

    # controls
    us = np.zeros((2, arm.N))

    # # states
    # xs = np.zeros((4, arm.N+1))
    # xs[:, 0] = x0

    # for k in range(arm.N):
    #     xs[:, k+1], _, _ = arm.f(k, xs[:, k], us[:, k])
    xs = arm.sys_traj(us)

    lb = np.concatenate((np.ones(arm.N)*-1.0, np.ones(arm.N)*-0.4))
    ub = np.concatenate((np.ones(arm.N)*1.0, np.ones(arm.N)*0.4))

    res = least_squares(arm.arm_cost, us.flatten(), bounds=(lb, ub))
    us = np.reshape(res.x, (arm.m, arm.N))
    xs = arm.sys_traj(us)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(xs[0, :], xs[1, :], '-b')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[1].plot(np.arange(0, arm.tf, arm.h), us[0, :])
    axs[1].plot(np.arange(0, arm.tf, arm.h), us[1, :])
    axs[1].set_xlabel('sec.')
    axs[1].legend(["u_0", "u_1"])
    plt.subplots_adjust(wspace=0.25)

    y = arm.arm_cost(us)
    J = np.transpose(y)@y / 2.0
    print("cost = ", J)
    plt.show()
