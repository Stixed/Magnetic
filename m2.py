# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:15:00 2018

@author: s.Kolodny
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


E = np.identity(3)  # Unit matrix definition
a1 = 3e-16
t1 = 10e-3
mu0 = 1.257e-6
Kc1 = -610.0  # First-order cubis anisotropy constant
gd = 1000# Gilbert damping
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

def det0(w,k,h):
    L = (1j*w/(Y*Ms))*E
    Heff=E*a1*(1j*k)**2+Na-(Z0+Zk)*E
    R=np.dot(Zx,Heff)
    RelaxAd=np.dot(Zx,R)
    T=L-R-(gd/Y)*RelaxAd
    Ti=np.linalg.inv(T)
    #h=np.dot(Zx,h)
    mv=np.dot(Ti,h)
#    print(mv[dem])
    return mv

def dispersion(k_min,k_max,w_min,w_max):
    k=np.linspace(k_min,k_max,500)
    w=np.linspace(w_min,w_max,500)
    w_k_x = np.zeros((500,500), dtype=complex)
    w_k_y = np.zeros((500, 500), dtype=complex)
    w_k_z = np.zeros((500, 500), dtype=complex)
    h=[0,1,0]
    for i in range(500):
        for j in range(500):
            mv=det0(w[i],k[j],h)
            w_k_x[i,j] = mv[0]
            w_k_y[i,j] = mv[1]
            w_k_z[i,j] = mv[2]
    plt.figure(1)
    plt.title("dispersion")
    plt.subplot(3,1,1)
    plt.pcolormesh(k, w, w_k_x.real)
    plt.subplot(3,1,2)
    plt.pcolormesh(k, w, w_k_y.real)
    plt.subplot(3,1,3)
    plt.pcolormesh(k, w, w_k_z.real)
    plt.show()
    return


def fourier(rt,n):
    m_x=np.zeros(np.size(t2),dtype=complex)
    m_y = np.zeros(np.size(t2),dtype=complex)
    m_z = np.zeros(np.size(t2),dtype=complex)
    h_t1=np.zeros((np.size(t2)),dtype=complex)
    for z in range(n):
        h_t1+=4000*np.exp(-(((t2-rt*z-30))**2)/1)
    h_w1=np.fft.fft(h_t1)
    w1=np.fft.fftfreq(h_w1.size)
    print(w1)

    for i in range(int(np.size(t2))):
        if i==0:
            i+=1
        mv=det0(w1[i]*1e11,4.5e6,[0,h_w1[i],0])
        m_x[i]=mv[0]
        m_y[i]=mv[1]
        m_z[i]=mv[2]

    m_x_t=np.fft.ifft(m_x)
    m_y_t = np.fft.ifft(m_y)
    m_z_t = np.fft.ifft(m_z)
    m_x_i=interp1d(t2,m_x_t.real,kind='cubic')
    m_y_i = interp1d(t2, m_y_t.real, kind='cubic')
    tnew=np.linspace(0, 9999, num=100000, endpoint=True)
    plt.figure(2)
    plt.title("precession")
    plt.plot(t2,h_t1.real)
    plt.show()
    plt.figure(3)
    plt.title("precession")
    plt.plot(m_x_i(tnew), m_y_i(tnew))
    plt.plot(0,0,'.')
    plt.show()
    plt.figure(4)
    plt.title("precession")
    plt.plot(m_x_t, m_y_t)
    plt.plot(0,0,'o')
    plt.show()
    return
#
#def field():
#    x_c = open('xcomp_180.txt', 'r')
#    y_c = open('ycomp_180.txt', 'r')
#    z_c = open('zcomp_180.txt', 'r')
#    l1 = [line.strip() for line in x_c]
#    l2 = [line.strip() for line in y_c]
#    l3 = [line.strip() for line in z_c]
#    Ef = np.zeros((4, 21), dtype=complex)
#    Hf = np.zeros((4, 21), dtype=complex)
#    for i in range(21):
#        temp1 = l1[i].split()
#        temp2 = l2[i].split()
#        temp3 = l3[i].split()
#        Ef[0, i] = float(temp1[0])
#        Hf[0, i] = float(temp1[0])
#        Ef[1, i] = float(temp1[1]) + 1j * float(temp1[2])
#        Ef[2, i] = 0
#        # Ef[2, i] = float(temp2[1]) + 1j * float(temp2[2])
#        Ef[3, i] = float(temp3[1]) + 1j * float(temp3[2])
#        print(Ef[1:, i])
#        Hf[1:, i] = -(1j) * np.cross(Ef[1:, i], np.conj(Ef[1:, i]))
#    print(Hf)
#    return Hf

def field():
    txt = open('Ex.txt', 'r')

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
            E=[float(temp1[3])+1j*float(temp1[6]),0,float(temp1[5])+1j*float(temp1[8])]
            H=-(1j/16*np.pi)*np.cross(E,np.conj(E))
            Hx[int(abs(temp1[0])/dx),:,int(abs(temp1[2])/dz)]=H[0]
            Hy[int(abs(temp1[0])/dx),:,int(abs(temp1[2])/dz)]=H[1]
            Hz[int(abs(temp1[0])/dx),:,int(abs(temp1[2])/dz)]=H[2]
print(Hz)
return Hx,Hy,Hz

dispersion(1e6,1e8,20e9,50e9)
fourier(2000,3)
