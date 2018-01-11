from sympy.solvers import solve as slv
from sympy import Symbol, Matrix, Function
import scipy as sc
import math as m
from scipy import optimize
from scipy.interpolate import interp1d
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

E = np.identity(3)  # Unit matrix definition
a1 = 3e-16
t1 = 10e-3
mu0 = 1.257e-6
Kc1 = -610.0  # First-order cubis anisotropy constant
gd = 0.1# Gilbert damping
z = np.array([0, 0, 1.0])
B = np.array([0, 0, 1.0])
M0 = np.array([0, 0, 140000.0])
H0 = B / mu0
mv = np.array([0, 0, 0])
H0s = np.sqrt(H0[0]**2+H0[1]**2+H0[2]**2)
Ms = np.sqrt(M0[0]**2+M0[1]**2+M0[2]**2)
Z0 = H0s / Ms
H0k = -(4 * Kc1 / (mu0 * Ms)) * np.array([0, 0, 1.0])
Zk = np.dot(H0k, z) / Ms
Zx = M0x = np.array([[0, -z[2], z[1]],
                     [z[2], 0, -z[0]],
                     [-z[1], z[0], 0]])
Na = -(4 * Kc1 / (mu0 * Ms ** 2)) * np.tensordot(z, z, axes=0)
Y=28e9*mu0
wM=Y*Ms
t2=np.linspace(0,9999,10000)

def fmap( rt, n):
    Hxw = np.zeros(np.size(t2),dtype=complex)
    Hxt = np.zeros(np.size(t2), dtype=complex)
    for z in range(n):
        Hxt += 1 * np.exp(-(((t2 - rt * z - 50)) ** 2) / 10)
    Hxw=np.fft.fft(Hxt)

    w1 = np.fft.fftfreq(Hxw.size)
    plt.figure(4)
    plt.title("precession")
    plt.subplot(2,1,1)
    plt.plot(w1, Hxw)
    plt.xlim(-0.05,0.05)
    plt.subplot(2,1,2)
    plt.plot(t2,Hxt)
    #plt.plot(0,0,'o')
    plt.show()

fmap(1000, 2)
