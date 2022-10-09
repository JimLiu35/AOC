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


def Riccati(t, cs, A, B, Q, R, xd):

    P = c2P(cs[0:3])
    s = cs[3:5]
    dP = -P@A - np.transpose(A)@P - Q + P@B@np.linalg.inv(R)@np.transpose(B)@P
    ds = - (np.transpose(A) - P@B@np.linalg.inv(R)@np.transpose(B))@s + Q@xd
    dcs = np.concatenate((P2c(dP), ds))
    return dcs


if __name__ == '__main__':

    Pf = np.diag((2, 0))

    A = np.array([[0, 1], [2, -1]])

    B = np.array([[0], [1]])

    Q = np.diag([2, 0])

    R = np.array([[0.005]])

    tf = 5
    dt = 0.1

    xd = np.array([1, 0])

    sf = - Pf@xd

    csf = np.concatenate((P2c(Pf), sf))

    def Riccati_(tf_, csf_): return Riccati(tf_, csf_, A, B, Q, R, xd)
    result = solve_ivp(Riccati_, (tf, 0), csf, method='RK45',
                       t_eval=np.arange(tf, 0, -dt))

    ts = np.flip(result.t)
    css = np.flip(result.y, axis=1)

    N = css.shape[1]

    cs = css[0:3, :]
    ss = css[3:5, :]

    x0 = np.array([-4, 4])

    fig, axs = plt.subplots(1, 3)
    axs[0].plot(ts, css[0, :])
    axs[0].plot(ts, css[1, :])
    axs[0].plot(ts, css[2, :])
    axs[0].legend(['P_{11}', 'P_{12}', 'P_{22}'])
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('P(t)')

    xs = np.zeros((2, N))
    us = np.zeros((1, N))

    x0 = np.array([-4, 4])
    xs[:, 0] = x0

    for i in range(N):
        P = c2P(cs[:, i])
        s = ss[:, i]
        l = P@xs[:, i] + s
        u = -np.linalg.inv(R)@np.transpose(B)@l
        K = A - B@np.linalg.inv(R)@np.transpose(B)@P
        if (i < N-1):
            xs[:, i+1] = xs[:, i] + dt*(A@xs[:, i] + B@u)
        us[:, i] = u

    axs[1].plot(ts, xs[0, :])
    axs[1].plot(ts, xs[1, :])
    axs[1].legend(['x_{1}', 'x_{2}'])
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('x(t)')

    axs[2].plot(ts, us[0, :])
    axs[2].legend(['u_{1}'])
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('u(t)')

    plt.subplots_adjust(wspace=0.6)
    plt.show()
