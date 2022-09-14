# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class HW1_Q3(object):
  """ Homework 1 Question 3 """


  def eval_L1(self, x):
    """Evaluate L(x)=(1-x1)^2+200*(x2-x1^2)"""
    """Input: x (numpy.ndarray, size: 2)"""
    """Output: L (numpy.float64)"""
    # 3a) TODO: Implement me
    x1 = x[0]
    x2 = x[1]
    L = (1-x[0])**2+200*(x[1]-x[0]**2)**2
    return L


  def eval_G1(self, x):
    """Evaluate gradient of L(x)=(1-x1)^2+200*(x2-x1^2)"""
    """Input: x (numpy.ndarray, size: 2)"""
    """Output: G (numpy.ndarray, size: 2)"""
    # 3a) TODO: Implement me
    G = np.zeros(2)
    x1 = x[0]
    x2 = x[1]
    G[0] = -800*x1*(-x1**2 + x2) + 2*x1 - 2
    G[1] = -400*x1**2 + 400*x2

    return G


  def eval_H1(self, x):
    """Evaluate hessian of L(x)=(1-x1)^2+200*(x2-x1^2)"""
    """Input: x (numpy.ndarray, size: 2)"""
    """Output: H (numpy.ndarray, size: 2x2)"""
    # 3a) TODO: Implement me
    H = np.zeros((2,2))
    x1 = x[0]
    x2 = x[1]
    H[0,0] = 2400*x1**2 - 800*x2 + 2
    H[0,1] = -800*x1
    H[1,0] = -800*x1
    H[1,1] = 400
    return H


  def armijo(self, x, fx, gx, d):
    """Calculate step size using armijo's rule"""
    """Inputs: x (numpy.ndarray, size: 2)
               fx (loss function)
               gx (gradient function)
               d (descent direction, numpy.ndarray, size: 2) """
    """Output: step (numpy.float64)"""
    # 3b) TODO: Implement me

    sigma = 0.01
    beta = 0.25
    s = 1

    f = fx(x)
    G = gx(x)
    a = beta*s
    for m in range(100):
        a = beta**m*s
        if (f - fx(x + a*d) > -sigma*beta**m*s*np.dot(G, d)):
            return a
    print(a)
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
    counter = 1
    # descent direction
    d = -gx(x)
    # get stepsize
    a = self.armijo(x,fx,gx,d)
    while (a >= 1e-8 and counter <= 500000):
        #print("stepnumber: {}".format(counter),end="\r")

        x =  x + a*d
        xs.append(x)
        # update d and a
        # descent direction
        d = -gx(x)
        # get stepsize
        a = self.armijo(x,fx,gx,d)

        counter = counter + 1
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
    counter = 1
    # descent direction
    d = np.linalg.solve(-hx(x),gx(x))
    # stepsize
    a = self.armijo(x,fx,gx,d)
    while (a >= 1e-8 and counter <= 500000):
        #print("stepnumber: {}".format(counter),end="\r")
        x =  x + a*d
        xs.append(x)
        # update d and a
        # descent direction
        
        d = np.linalg.solve(-hx(x),gx(x))

        # get stepsize
        a = self.armijo(x,fx,gx,d)
        counter = counter + 1
        # check for positive definiteness
        e = np.linalg.eigvals(hx(x))
        if (np.amin(e) < 0):
            print("Hessian matrix is not positive definitness!")
            return np.array(xs)
    return np.array(xs)



  def eval_L2(self, x):
    """Evaluate L(x) = x1*exp(-x1^2-(1/2)*x2^2) + x1^2/10 + x2^2/10"""
    """Input: x (numpy.ndarray, size: 2)"""
    """Output: L (numpy.float64)"""
    x1 = x[0]
    x2 = x[1]
    L = x[0]*np.exp(-(x[0]**2 + 0.5*x[1]**2)) + (x[0]**2 + x[1]**2)/10
    return L



  def eval_G2(self, x):
    """Evaluate gradient of L(x) = x1*exp(-x1^2-(1/2)*x2^2) + x1^2/10 + x2^2/10"""
    """Input: x (numpy.ndarray, size: 2)"""
    """Output: G (numpy.ndarray, size: 2)"""
    G = np.zeros(2)
    x1 = x[0]
    x2 = x[1]
    G[0] = -2*x1**2*np.exp(-x1**2 - 0.5*x2**2) + x1/5 + np.exp(-x1**2 - 0.5*x2**2)
    G[1] = -1.0*x1*x2*np.exp(-x1**2 - 0.5*x2**2) + x2/5

    return G


  def eval_H2(self, x):
    """Evaluate hessian of L(x) = x1*exp(-x1^2-(1/2)*x2^2) + x1^2/10 + x2^2/10"""
    """Input: x (numpy.ndarray, size: 2)"""
    """Output: H (numpy.ndarray, size: 2x2)"""
    H = np.zeros((2,2))
    x1 = x[0]
    x2 = x[1]
    H[0,0] = 4*x1**3*np.exp(-x1**2 - 0.5*x2**2) - 6*x1*np.exp(-x1**2 - 0.5*x2**2) + 1/5
    H[0,1] = 2.0*x1**2*x2*np.exp(-x1**2 - 0.5*x2**2) - 1.0*x2*np.exp(-x1**2 - 0.5*x2**2)
    H[1,0] = 2.0*x1**2*x2*np.exp(-x1**2 - 0.5*x2**2) - 1.0*x2*np.exp(-x1**2 - 0.5*x2**2)
    H[1,1] = 1.0*x1*x2**2*np.exp(-x1**2 - 0.5*x2**2) - 1.0*x1*np.exp(-x1**2 - 0.5*x2**2) + 1/5
    return H


if __name__ == '__main__':
    """This code runs if you execute this script"""
    hw1_q3 = HW1_Q3()

    # # TODO: Uncomment this line to visualize gradient descent & newton descent 
    # #       for first loss function
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x1 = np.arange(-2, 2, 0.01)
    x2 = np.arange(-2, 2, 0.01)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros((X1.shape[0], X1.shape[1]))
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = hw1_q3.eval_L1(np.array([X1[i, j], X2[i, j]]))
    contour = ax.contour(X1, X2, Z, cmap=cm.viridis)
    xsg = hw1_q3.gradient_descent(np.array([0, 0]), hw1_q3.eval_L1,hw1_q3.eval_G1)
    xsn = hw1_q3.newton_descent(np.array([0, 0]), hw1_q3.eval_L1, hw1_q3.eval_G1, hw1_q3.eval_H1)
    plt.plot(xsg[:, 0], xsg[:, 1], '-o')
    plt.plot(xsn[:, 0], xsn[:, 1], '-o')
    print("First loss function gradient method steps: ", xsg.shape[0]-1)
    print("First loss function newton method steps:   ", xsn.shape[0]-1)
    
    # # TODO: Uncomment this line to visualize gradient descent & newton descent 
    # #       for second loss function
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
    xsn = hw1_q3.newton_descent(np.array([1.5, 3.0]), hw1_q3.eval_L2, 
        hw1_q3.eval_G2, hw1_q3.eval_H2)
    plt.plot(xsg[:, 0], xsg[:, 1], '-o')
    plt.plot(xsn[:, 0], xsn[:, 1], '-o')
    print("Second loss function gradient method steps: ", xsg.shape[0]-1)
    print("Second loss function newton method steps:   ", xsn.shape[0]-1)


    plt.show()

    

