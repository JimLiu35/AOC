# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class HW1_Q3(object):
    """ Homework 1 Question 3 """

    def eval_L1(self, x):
        """Evaluate L(x)=(1-x1)^2+200*(x2-x1^2)^2"""
        """Input: x (numpy.ndarray, size: 2)"""
        """Output: L (numpy.float64)"""
        # 3a) TODO: Implement me

        L = 0.0
        L = (1 - x[0])**2 + 200 * (x[1] - x[0]**2)**2
        return L

    def eval_G1(self, x):
        """Evaluate gradient of L(x)=(1-x1)^2+200*(x2-x1^2)"""
        """Input: x (numpy.ndarray, size: 2)"""
        """Output: G (numpy.ndarray, size: 2)"""
        # 3a) TODO: Implement me

        G = np.zeros(2)
        G[0] = 2*x[0] - 2 - 800*x[0]*x[1] + 800 * x[0]**3
        G[1] = 400 * x[1] - 400 * x[0]**2
        return G

    def eval_H1(self, x):
        """Evaluate hessian of L(x)=(1-x1)^2+200*(x2-x1^2)"""
        """Input: x (numpy.ndarray, size: 2)"""
        """Output: H (numpy.ndarray, size: 2x2)"""
        # 3a) TODO: Implement me

        H = np.zeros((2, 2))
        H[0][0] = 2 - 800 * x[1] + 2400 * x[0]**2
        H[0][1] = -800 * x[0]
        H[1][0] = -800 * x[0]
        H[1][1] = 400
        return H

    def armijo(self, x, fx, gx, d):
        """Calculate step size using armijo's rule"""
        """Inputs: x (numpy.ndarray, size: 2)
               fx (loss function)
               gx (gradient function)
               d (descent direction, numpy.ndarray, size: 2) """
        """Output: step (numpy.float64)"""
        # 3b) TODO: Implement me

        a = 0.0
        s = 1.0             # Assume the initial step size is 1
        beta = 0.25         # Assume the rate of decrease is 0.25
        sigma = 0.01        # Choose the acceptance ratio is 0.01
        m = 0
        for i in range(100):
            # Actual descent
            aDescent = fx(x) - fx(x + beta**m * s * d)
            # Predict descent
            pDescent = -sigma * beta**m * s * np.vdot(gx(x), d)
            if aDescent >= pDescent:
                break
            else:
                m = m + 1
        a = beta ** m * s
        return a

    def gradient_descent(self, x0, fx, gx):
        """Perform gradient descent"""
        """Inputs: x0 (numpy.ndarray, size: 2), 
               fx (loss function), 
               gx (gradient function)"""
        """Outputs: xs (numpy.ndarray, size: (numSteps+1)x2)"""
        # 3c) TODO: Implement me

        x = x0
        xs = [x]
        for i in range(500000):
            d = -gx(x)           # Direction
            if np.linalg.norm(d) < 1e-8:
                return np.array(xs)
            a = self.armijo(x, fx, gx, d)
            x = x + a * d
            xs.append(x)
            print(i, end="\r")

        return np.array(xs)

    def newton_descent(self, x0, fx, gx, hx):
        """Perform gradient descent"""
        """Inputs: x0 (numpy.ndarray, size: 2), 
               fx (loss function), 
               gx (gradient function)
               hx (hessian function)"""
        """Outputs: xs (numpy.ndarray, size: (numSteps+1)x2)"""
        # 3d) TODO: Implement me

        x = x0
        xs = [x]
        for i in range(500000):
            gradient = gx(x)
            Hessian = hx(x)

            if np.linalg.norm(gradient) < 1e-8:
                return np.array(xs)

            d = -np.dot(np.linalg.inv(Hessian), gradient)
            a = self.armijo(x, fx, gx, d)
            x = x + a * d
            xs.append(x)
            print(i, end="\r")

        return np.array(xs)

    def eval_L2(self, x):
        """Evaluate L(x) = x1*exp(-x1^2-(1/2)*x2^2) + x1^2/10 + x2^2/10"""
        """Input: x (numpy.ndarray, size: 2)"""
        """Output: L (numpy.float64)"""
        L = 0.0
        x1 = x[0]
        x2 = x[1]
        L = x1 * np.exp(-x1**2 - 0.5 * x2**2) + x1**2 / 10 + x2**2 / 10
        return L

    def eval_G2(self, x):
        """Evaluate gradient of L(x) = x1*exp(-x1^2-(1/2)*x2^2) + x1^2/10 + x2^2/10"""
        """Input: x (numpy.ndarray, size: 2)"""
        """Output: G (numpy.ndarray, size: 2)"""

        G = np.zeros(2)
        x1 = x[0]
        x2 = x[1]
        G[0] = np.exp(-x1**2 - 0.5 * x2**2) - 2 * x1**2 * \
            np.exp(-x1**2 - 0.5 * x2**2) + x1 / 5
        G[1] = -x1 * x2 * np.exp(-x1**2 - 0.5 * x2**2) + x2 / 5
        return G

    def eval_H2(self, x):
        """Evaluate hessian of L(x) = x1*exp(-x1^2-(1/2)*x2^2) + x1^2/10 + x2^2/10"""
        """Input: x (numpy.ndarray, size: 2)"""
        """Output: H (numpy.ndarray, size: 2x2)"""

        H = np.zeros((2, 2))
        x1 = x[0]
        x2 = x[1]
        H[0][0] = 4 * x1**3 * np.exp(-x1**2 - 0.5 * x2**2) - 6 * x1 * \
            np.exp(-x1**2 - 0.5 * x2**2) + 1 / 5
        H[0][1] = 2 * x1**2 * x2 * np.exp(-x1**2 - 0.5 * x2**2) - x2 * \
            np.exp(-x1**2 - 0.5 * x2**2)
        H[1][0] = 2 * x1**2 * x2 * np.exp(-x1**2 - 0.5 * x2**2) - x2 * \
            np.exp(-x1**2 - 0.5 * x2**2)
        H[1][1] = x1 * x2**2 * np.exp(-x1**2 - 0.5 * x2**2) - x1 * \
            np.exp(-x1**2 - 0.5 * x2**2) + 1 / 5
        return H


if __name__ == '__main__':
    """This code runs if you execute this script"""
    hw1_q3 = HW1_Q3()

    # TODO: Uncomment the following code to visualize gradient descent
    #       & newton's method for the first loss function
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # x1 = np.arange(-2, 2, 0.01)
    # x2 = np.arange(-2, 2, 0.01)
    # X1, X2 = np.meshgrid(x1, x2)
    # Z = np.zeros((X1.shape[0], X1.shape[1]))
    # for i in range(X1.shape[0]):
    #     for j in range(X1.shape[1]):
    #         Z[i, j] = hw1_q3.eval_L1(np.array([X1[i, j], X2[i, j]]))
    # contour = ax.contour(X1, X2, Z, cmap=cm.viridis)
    # xsg = hw1_q3.gradient_descent(np.array([0, 0]), hw1_q3.eval_L1,
    #                               hw1_q3.eval_G1)
    # xsn = hw1_q3.newton_descent(np.array([0, 0]), hw1_q3.eval_L1,
    #                             hw1_q3.eval_G1, hw1_q3.eval_H1)
    # plt.plot(xsg[:, 0], xsg[:, 1], '-o')
    # plt.plot(xsn[:, 0], xsn[:, 1], '-o')
    # print("First loss function gradient method steps: ", xsg.shape[0]-1)
    # print("First loss function newton method steps:   ", xsn.shape[0]-1)

    # TODO: Uncomment the following code to visualize gradient descent
    #       & newton's method for the second loss function
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x1 = np.arange(-3.5, 3.5, 0.01)
    x2 = np.arange(-3.5, 3.5, 0.01)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros((X1.shape[0], X1.shape[1]))
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = hw1_q3.eval_L2(np.array([X1[i, j], X2[i, j]]))
    contour = ax.contour(X1, X2, Z, cmap=cm.viridis)
    xsg = hw1_q3.gradient_descent(np.array([1.5, 3.0]), hw1_q3.eval_L2,
                                  hw1_q3.eval_G2)
    print("-"*50)
    xsn = hw1_q3.newton_descent(np.array([1.5, 3.0]), hw1_q3.eval_L2,
                                hw1_q3.eval_G2, hw1_q3.eval_H2)  # Not quite working yet
    plt.plot(xsg[:, 0], xsg[:, 1], '-o')
    plt.plot(xsn[:, 0], xsn[:, 1], '-o')
    print("Second loss function gradient method steps: ", xsg.shape[0]-1)
    print("Second loss function newton method steps:   ", xsn.shape[0]-1)
    plt.show()
