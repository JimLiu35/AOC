# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from control import lqr


class HW4_Q3(object):
    """ Homework 4 Question 3 """

    def P2c(self, P):
        # convert from a 2x2 symmetric matrix to a 3x1 vector of its unique entries
        c = np.array([P[0, 0], P[0, 1], P[1, 1]])
        return c

    def c2P(self, c):
        # the reverse of P2c
        P = np.array([[c[0], c[1]], [c[1], c[2]]])
        return P

    def __init__(self):
        """    
        A: numpy.ndarray, shape: (2,2)
        B: numpy.ndarray, shape: (2,1)
        Q: numpy.ndarray, shape: (2,2)
        R: numpy.ndarray, shape: (1,1)
        tf: float
        Pf: numpy.ndarray, shape: (2,2)
        dt: float
        x0: numpy.ndarray, shape: (2,)
        """
        # 3a) TODO: Plug in A, B, Q, R, tf, Pf, dt, x0

        self.A = np.zeros((2, 2))
        self.B = np.zeros((2, 1))
        self.Q = np.zeros((2, 2))
        self.R = np.zeros((1, 1))
        self.tf = 20
        self.Pf = np.zeros((2, 2))
        self.dt = 0.1
        self.x0 = np.zeros(2)

        self.A[0, 0] = 0
        self.A[0, 1] = 1
        self.A[1, 0] = 2
        self.A[1, 1] = -1

        self.B[0] = 0
        self.B[1] = 1

        self.Q[0, 0] = 1
        self.Q[0, 1] = 0
        self.Q[1, 0] = 0
        self.Q[1, 1] = 0.5

        self.R = 0.5
        self.x0[0] = -5
        self.x0[1] = 5

