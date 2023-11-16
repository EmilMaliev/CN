# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 22:37:04 2022

@author: Professional
"""

import numpy as np
import math as m

def k(x, s):
    return np.exp(-x - s)

def k1(x, s):
    return np.exp(-x - s) * (s <= x)

def f(x):
    return 0.5 * (np.exp(-x) + np.exp(-3 * x))


def value_in_point(value, K, f, method, equ_type, n=10, N=10, l=0.3, a=0., b=1.):
    h = (b - a) / N
    x = np.arange(a, b + h, h)
    summands = np.zeros(N + 1)
    y, coefficients = fixed_point_iteration(f, k, method, equ_type, N, n, l, a, b)
    for i in range(N + 1):
        summands[i] = coefficients[i] * y[i] * K(value, x[i])
    return l * sum(summands) + f(value)


def fixed_point_iteration(f, k, method, equ_type, N=10, n=10, l=0.3, a=0., b=1.):
    h = (b - a) / N
    x = np.array([a + h * i for i in range(N + 1)])
    coefficients = np.full(shape=N + 1, fill_value=h)
    coefficients[0] = 0
    y = np.zeros(N + 1)

    if method == 'Medium Riemann rule':
        coefficients[0], coefficients[N] = h, h
        for i in range(N - 1):
            x[i] = x[i] + h/2 

    phi = np.zeros((n + 1, N + 1))

    for i in range(n + 1):
        for j in range(N + 1):
            if i == 0:
                phi[i][j] = f(x[j])
            else:
                phi[i][j] = np.sum([coefficients[m] * k(x[j], x[m])
                    * phi[i - 1][m] for m in range(N + 1)])

    for i in range(n + 1):
       for j in range(N + 1):
            y[j] += l**i * phi[i][j]
            

    return y, coefficients

def phi_fredholm(i, x):
    return (1 / (4 * (2 ** i))) * (np.exp(2) - 1) ** i * (1 + 3 * np.exp(2)) * np.exp(-x - (2 + 2 * i))


def solution_fredholm(x, n, f, lmb):
    sol = f(x)
    for i in range(1, n + 1, 1):
        sol += phi_fredholm(i, x) * (lmb ** i)
    return sol


def phi_volterra(i, x):
    return (1 / m.factorial(i+1)) * np.exp(-x * (i+2)) * (np.sinh(x) ** i) * (i * np.sinh(x) + (i+1) * np.cosh(x))


def solution_volterra(x, n, f, lmb):
    sol = f(x)
    for i in range(1, n + 1, 1):
        sol += phi_volterra(i, x) * (lmb ** i)
    return sol

x = np.arange(0., 1. + .1, .1)

print('Fredholm in nodes')
for i in range(len(x)):
    print("x", i, 
        value_in_point(x[i], k, f, 'Right Riemann sum', 'Fredholm', l=0.3, N=10),
        solution_fredholm(x[i], 100, f, 0.3),
        solution_fredholm(x[i], 100, f, 0.3) - value_in_point(x[i], k, f, 'Right Riemann sum', 'Fredholm', l=0.3, N=10)
        )
print('Fredholm in 1/2.2')
print   (
        value_in_point(1/2.2, k, f, 'Right Riemann sum', 'Fredholm', l=0.3, N=10),
        solution_fredholm(1/2.2, 100, f, 0.3),
        solution_fredholm(1/2.2, 100, f, 0.3) - value_in_point(1/2.2, k, f, 'Right Riemann sum', 'Fredholm', l=0.3, N=10)
        )

print()

print('Volterra in nodes')
for i in range(len(x)):
    print("x", i,
        value_in_point(x[i], k1, f, 'Medium Riemann rule', 'Volterra', l=0.3, N=10),
        solution_volterra(x[i], 100, f, 0.3),
        solution_volterra(x[i], 100, f, 0.3) - value_in_point(x[i], k1, f, 'Medium Riemann rule', 'Volterra', l=0.3, N=10)
        )
print('Volterra in 1/2.2')
print   (
        value_in_point(1/2.2, k1, f, 'Medium Riemann rule', 'Volterra', l=0.3, N=10),
        solution_volterra(1/2.2, 100, f, 0.3),
        solution_volterra(1/2.2, 100, f, 0.3) - value_in_point(1/2.2, k1, f, 'Medium Riemann rule', 'Volterra', l=0.3, N=10)
        )


q = 0.3 
print((0.3 ** 6) / 1.4)
print((q ** 6 / ((1 - (q / 6)) * np.math.factorial(6))) * 0.5)
points = np.arange(0, 1., 0.1)
