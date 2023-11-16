# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:14:02 2022

@author: Professional
"""

from scipy import integrate
def f(x):
    return (2**x)/(1 - 4**x)
v, err = integrate.quad(f, -2, -1)
print ('Точное вычичление интеграла = ', v)

e = 0.00001
Q2 = 0
h = 1
Q1 = f(-2)
N = 1
while abs((Q1 - Q2) * 2) > e:
    h = h / 2
    Q2 = Q1
    N *= 2
    Q1 = 0
    for i in range(N):
        Q1 += f(-2 + h * i)
    Q1 *= h
print(h)
print(Q1)
print(v - Q1)
   
