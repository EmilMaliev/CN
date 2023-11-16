# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:34:17 2022

@author: Professional
"""

from scipy import integrate
def f(x):
    return (2**x)/(1 - 4**x)
v, err = integrate.quad(f, -2, -1)
print ('Точное вычичление интеграла = ', v)

Q = 0
for i in range(100):
    Q += f(-2 + i * 0.01 + 0.005)
Q *= 0.01

print(Q, v - Q)