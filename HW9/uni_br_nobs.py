import numpy as np
import sympy as sp

# EN530.603 computes nonlinear observability of unicycle with range-bearing
# from beacons
# M. Kobilarov , marin(at)jhu.edu


class Problem:

  def __init__(self):

    # # one beacon : unobservable
    # self.pbs = np.array([[5], [5]])    # beacon positions

    # two beacons : observable
    self.pbs = np.array([[5, 3], [5, 6]])    # beacon positions

    self.nb = self.pbs.shape[1]  # number of beacons

  def uni_f(self, x, u):
    # dynamical model of the unicycle
    c = sp.cos(x[2])
    s = sp.sin(x[2])

    f = sp.zeros(3, 1)
    f[0] = c * u[0]
    f[1] = s * u[0]
    f[2] = u[1]

    return f

  def br_h(self, x):

    p = sp.Matrix([x[0], x[1]])
    y = sp.zeros(self.nb * 2, 1)
    H = sp.zeros(self.nb * 2, 3)

    for i in range(self.nb):
      pb = self.pbs[:, i]  # i-th beacon
      d = sp.Matrix([pb[0] - p[0], pb[1] - p[1]])
      r = d.norm()
      th = sp.atan2(d[1], d[0]) - x[2]

      y[i * 2] = th
      y[i * 2 + 1] = r

      H[i * 2:i * 2 + 2, :] = sp.Matrix([[d[1] / r**2, -d[0] / r**2, -1],
                                        [-d[0] / r, -d[1] / r, 0]])

    return y, H


prob = Problem()

px, py, th, u1, u2 = sp.symbols('px,py,th,u1,u2', real=True)

x = sp.Matrix([[px], [py], [th]])
u = sp.Matrix([[u1], [u2]])

f = prob.uni_f(x, u)
z, H = prob.br_h(x)
f = sp.simplify(f)
z = sp.simplify(z)
H = sp.simplify(H)

nz = len(z)
nx = len(x)

dz = H @ f
l = sp.zeros(nz + len(dz), 1)
for i in range(nz):
  l[i] = z[i]
for i in range(len(dz)):
  l[i + nz] = dz[i]
l = sp.simplify(l)

for i in range(3):
  Dl = l.jacobian(x)

  A = Dl.subs(px, 1).subs(py, 2).subs(th, np.pi / 4)

  # # try symbolic rank and also with different values for controls
  # # TODO: SYMPY IS SOMETIMES OUTPUTTING AN INCORRECT RANK
  # rs = [A.evalf().rank(),
  #       A.subs(u1, 1).subs(u2, 1).rank(),
  #       A.subs(u1, 1).subs(u2, 0).rank(),
  #       A.subs(u1, 0).subs(u2, 1).evalf().rank(),
  #       A.subs(u1, 0).subs(u2, 0).evalf().rank()]

  rs = [np.linalg.matrix_rank(
    np.array(A.subs(u1, 1).subs(u2, 1).evalf()).astype(np.float64)),
    np.linalg.matrix_rank(
    np.array(A.subs(u1, 1).subs(u2, 0).evalf()).astype(np.float64)),
    np.linalg.matrix_rank(
    np.array(A.subs(u1, 0).subs(u2, 1).evalf()).astype(np.float64)),
    np.linalg.matrix_rank(
    np.array(A.subs(u1, 0).subs(u2, 0).evalf()).astype(np.float64))]

  if nx in rs:
    # rank is 3
    break

  # keep adding time-derivatives of z
  dz = dz.jacobian(x) @ f
  l_ = sp.zeros(len(l) + len(dz), 1)
  for i in range(len(l)):
    l_[i] = l[i]
  for i in range(len(dz)):
    l_[i + len(l)] = dz[i]
  l = l_

if max(rs) == nx:
  print('observable')
else:
  print('unobservable')
