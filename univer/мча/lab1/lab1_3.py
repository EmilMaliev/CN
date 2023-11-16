# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:19:33 2022

@author: Professional
"""

import numpy as np
import scipy as sp
import math as m
from scipy import integrate
from scipy.misc import derivative

def f(x):
    return (2**x)/(1 - 4**x)
v, err = integrate.quad(f, -2, -1)
print ('Точное вычичление интеграла = ', v)

def gaussian_quadrature(f, n = 5.0, a = -2.0, b = -1.0):
    roots = sp.special.legendre(n).roots
    summands = map(lambda x : (f((a + b) * 0.5 + x * (b - a) * 0.5) /
((1.0 - x ** 2.0) * np.polyval(sp.special.legendre(n).deriv(1), x) ** 2.0)), roots)
    return (b - a) * sum(summands)

def error(max_deriviative = 29085, n = 5):
    return (max_deriviative * 2 ** (2 * n + 3) / ((2 * n + 3) * m.factorial(2 * n + 2))) * ((m.factorial(n + 1)) ** 2 / m.factorial(2 * n + 2)) ** 2

print(error())
I = gaussian_quadrature(f)
print(I - sp.integrate.quad(lambda x : f(x), -2.0, -1.0))

