#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:42:07 2017

@author: stixed
"""

import scipy as sc
import math as m
import numpy as np
import matplotlib.pyplot as plt


E=np.identity(3) # Unit matrix definition
a1=0.049
K1=-5.421e-4*1e-6 #J/cm^3
mu0=1.257e-6
h=1.0545718e-34
#alpha=a1*E
B=[0,0,1.0]
M0=[0,0,0.1]
Hi=B # Vector M0
Hi[0]= B[0]/mu0 # H_i vector
Hi[1]= B[1]/mu0
Hi[2]= B[2]/mu0
n=[0,0,1.0] #normal vector
 # beta
g=2*mu0/h
M0[0]= M0[0]/mu0 # H_i vector
M0[1]= M0[1]/mu0
M0[2]= M0[2]/mu0
M0l=m.sqrt(M0[0]**2+M0[1]**2+M0[2]**2)
b=2*K1/(M0l**2)


print(M0)
print(Hi)
k_n=np.linspace(0,100-1,100)
w_k=np.zeros((100),dtype=np.complex)

def det0(w,k):
    M0l=m.sqrt(M0[0]**2+M0[1]**2+M0[2]**2) # the length of M0 vector
    a=np.dot(M0,Hi)
    a2=np.dot(M0,n) # dot production of M0*H_i and M_0*n
    x1=M0[1]*n[2]-M0[2]*n[1]
    x2=M0[2]*n[0]-M0[0]*n[2]
    x3=M0[0]*n[1]-M0[1]*n[0]# vector production
    Add=np.array([[n[0]*b*x1,n[1]*b*x1,n[2]*b*x1],
              [n[0]*b*x2,n[1]*b*x2,n[2]*b*x2],
              [n[0]*b*x3,n[1]*b*x3,n[2]*b*x3]]) # additional matrix of (M0x(n)*(m*n))
    
    X=(1j*w/g)*E
    #alpha=(1j*k)**2*a1*E
    S=(1j*k)**2*a1*E-((1/M0l**2.0)*(a+b*a2**2.0))*E
    S_f=np.array([[M0[1]*S[2][0],M0[1]*S[2][1],M0[1]*S[2][2]],
                  [M0[2]*S[0][0],M0[2]*S[0][1],M0[2]*S[0][2]],
                  [M0[0]*S[1][0],M0[0]*S[1][1],M0[0]*S[1][2]]])
    S_r=np.array([[M0[2]*S[1][0],M0[2]*S[1][1],M0[2]*S[1][2]],
                  [M0[0]*S[2][0],M0[0]*S[2][1],M0[0]*S[2][2]],
                  [M0[1]*S[0][0],M0[1]*S[0][1],M0[1]*S[0][2]]])
    S_s=X-S_f+S_r-Add
    #print(S_s)
    #return np.real(np.linalg.det(S_s))
    return np.abs(np.linalg.det(S_s)*10.0**90)
def disp(k):
    value=sc.optimize.fsolve(det0,k,args=k,full_output=False)
    return value[0]

for i in range(100):
    r=i*1e-6
    w_k[i]=disp(r)
    #print(w_k[i])
plt.title("w(k)")
plt.plot(k_n, w_k, label="w")
#plt.plot(k_n, w_k.imag, label="w_imag")
plt.legend(loc='upper left')
plt.show()
plt.clf()
plt.close()
