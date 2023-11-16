# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:17:49 2022

@author: Professional
"""

from scipy import integrate
def f(x):
    return (2**x)/(1 - 4**x)
v, err = integrate.quad(f, -2, -1)
print ('Точное вычичление интеграла = ', v)

Q = (0.01 * (f(-2) + f(-1))) / 2 

for i in range(1, 100):
    Q += f(-2 + i * 0.01)
Q *= 0.01
    
print(Q, v - Q)