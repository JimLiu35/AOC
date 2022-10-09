import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def P2c(P):
    # convert from a 2x2 symmetric matrix to a 3x1 vector of its unique entries
    c = np.array([P[0, 0], P[0, 1], P[1, 1]])
    return c


def c2P(c):
    # the reverse of P2c
    P = np.array([[c[0], c[1]], [c[1], c[2]]])
    return P


def Riccati(t, c, A, B, Q, R):

    P = c2P(c)
    dP = -P@A - np.transpose(A)@P - Q + P@B@np.linalg.inv(R)@np.transpose(B)@P
    dc = P2c(dP)
    return dc


if __name__ == '__main__':

    Pf = np.zeros((2, 2))

    A = np.array([[0, 1], [2, -1]])

    B = np.array([[0], [1]])

    Q = np.diag([1, 0.5])

    R = np.array([[0.25]])

    tf = 5

    dt = 0.1

    cf = P2c(Pf)

    def Riccati_(tf_, cf_): return Riccati(tf_, cf_, A, B, Q, R)
    result = solve_ivp(Riccati_, (tf, 0), cf, method='RK45',
                       t_eval=np.arange(tf, 0, -dt))

    ts = np.flip(result.t)
    cs = np.flip(result.y, axis=1)

    N = cs.shape[1]

    fig, axs = plt.subplots(1, 3)
    axs[0].plot(ts, cs[0, :])
    axs[0].plot(ts, cs[1, :])
    axs[0].plot(ts, cs[2, :])
    axs[0].legend(['P_{11}', 'P_{12}', 'P_{22}'])
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('P(t)')

    xs = np.zeros((2, N))
    us = np.zeros((1, N))

    x0 = np.array([-4, 4])
    xs[:, 0] = x0
    Ps = c2P(cs)
    print(Ps)
    # print(N)

    for i in range(N):
        P = c2P(cs[:, i])
        l = P@xs[:, i]
        u = -np.linalg.inv(R)@np.transpose(B)@l
        K = A - B@np.linalg.inv(R)@np.transpose(B)@P
        if (i < N-1):
            xs[:, i+1] = xs[:, i] + dt*K@xs[:, i]
        us[:, i] = u

    axs[1].plot(ts, us[0, :])
    axs[1].legend(['u_{1}'])
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('u(t)')

    axs[2].plot(ts, xs[0, :])
    axs[2].plot(ts, xs[1, :])
    axs[2].legend(['x_{1}', 'x_{2}'])
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('x(t)')

    plt.subplots_adjust(wspace=0.6)
    plt.show()
