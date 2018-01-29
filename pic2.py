#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:39:12 2018

@author: stixe
"""

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
w_r=4.56e9
dt=1e-14 #seconds
Fs=1/dt
rts=1/w_r
rt=int(rts/dt)
sw=50e-15
swd=sw/dt
n=10

E = np.identity(3)  # Unit matrix definition
a1 = 3e-16
t1 = 10e-3
mu0 = 1.257e-6
Kc1 = -610.0  # First-order cubis anisotropy constant
gd = 2.75e3# Gilbert damping
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
t2=np.linspace(0,rt*n-1,rt*n)

dx=5.0
dy=5.0
dz=5.0

def fmap( rt, n):
 # Hz
    z_c=np.zeros((21,21,3,np.size(t2)),dtype=complex)
    y_c=np.zeros((21,61,3,np.size(t2)),dtype=complex)
    z_c_w=np.zeros((21,21,3,np.size(t2)),dtype=complex)
    y_c_w=np.zeros((21,61,3,np.size(t2)),dtype=complex)
    w_p=np.fft.fftfreq(np.size(t2))
    m_w_z=np.zeros((21,21,3,np.size(t2)),dtype=complex)
    m_w_y=np.zeros((21,61,3,np.size(t2)),dtype=complex)
    m_t_z=np.zeros((21,21,3,np.size(t2)),dtype=complex)
    m_t_y=np.zeros((21,61,3,np.size(t2)),dtype=complex)
    Hx,Hy,Hz,x,y,z=field()
#    print(Hx)
    Hxm=np.amax(Hx)
    Hym=np.amax(Hy)
    Hzm=np.amax(Hz)
#    print(Hym)
#    Hx=Hx/Hxm
#    Hy=Hy/Hym
#    Hz=Hz/Hzm
    for i in range(21):
        for j in range(21):
            for z in range(n):
                z_c[i,j,0,:] += Hx[i,j,0] * np.exp(-(((t2 - rt * z)) ** 2) /swd)
                z_c[i,j,1,:] += Hy[i,j,0] * np.exp(-(((t2 - rt * z)) ** 2) /swd)
                z_c[i,j,2,:] += Hz[i,j,0] * np.exp(-(((t2 - rt * z)) ** 2) /swd)
            z_c_w[i,j,0,:]=np.fft.fft(z_c[i,j,0,:])
            z_c_w[i,j,1,:]=np.fft.fft(z_c[i,j,1,:])
            z_c_w[i,j,2,:]=np.fft.fft(z_c[i,j,2,:])
    plt.figure(1)
    plt.title("Spectrum of the gaussian train")
    plt.plot(w_p*Fs/(2*np.pi),np.abs(z_c_w[0,0,1,:]))
    plt.xlim(4e9,7e9)
    plt.show()
        
def field():
    txt = open('Ex.txt', 'r')
    x=np.zeros(21)
    y=np.zeros(21)
    z=np.zeros(61)

    Ex=np.zeros((21,21,61),dtype=complex)
    Ey=np.zeros((21,21,61),dtype=complex)
    Ez=np.zeros((21,21,61),dtype=complex)
    
    Hx=np.zeros((21,21,61),dtype=complex)
    Hy=np.zeros((21,21,61),dtype=complex)
    Hz=np.zeros((21,21,61),dtype=complex)
    l1 = [line.strip() for line in txt]
    for i in range(np.size(l1)):
        temp = l1[i].split()
        temp1=[float(i) for i in temp]
        if temp1[0]<=100 and temp1[0]>=0 and temp1[1]<=100 and temp1[1]>=0 and temp1[2]<=0 and temp1[2]>=-300:
            Ex[int(abs(temp1[0])/dx),:,int(abs(temp1[2])/dz)]=float(temp1[3])+1j*float(temp1[6])
            Ey[int(abs(temp1[0])/dx),:,int(abs(temp1[2])/dz)]=float(temp1[4])+1j*float(temp1[7])
            Ez[int(abs(temp1[0])/dx),:,int(abs(temp1[2])/dz)]=float(temp1[5])+1j*float(temp1[8])
            x[int(abs(temp1[0])/dx)]=temp[0]
            y[int(abs(temp1[1])/dy)]=temp[1]
            z[int(abs(temp1[2])/dz)]=temp[2]
            E=[float(temp1[3])+1j*float(temp1[6]),0,float(temp1[5])+1j*float(temp1[8])]
            H=-(1j/16*np.pi)*np.cross(E,np.conj(E))
#            print(H)
            Hx[int(abs(temp1[0])/dx),:,int(abs(temp1[2])/dz)]=H[0]
            Hy[int(abs(temp1[0])/dx),:,int(abs(temp1[2])/dz)]=H[1]
            Hz[int(abs(temp1[0])/dx),:,int(abs(temp1[2])/dz)]=H[2]
#    print(Hz)
    return Hx,Hy,Hz,x,y,z
fmap(rt,n)
