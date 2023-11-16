# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 22:38:09 2022

@author: Professional
"""

import sympy as sym
import numpy as np
from sympy import symbols


time, u, x = symbols('time u x')


def newton_method(f, x0=0., eps=10**-16):
    x1 = x0
    x2 = x0+1.0
    df = sym.diff(f, x)

    while np.abs(x1 - x2) >= eps:
        x2 = x1
        x1 = x1 - (f.subs(x, x1) / df.subs(x, x1))

    return x1


def euler_method(f, u0=1., partitions=10, a=0., b=1.):
    len = b-a
    tau = len / partitions
    t = np.zeros(partitions + 1)
    for i in range(partitions + 1):
        t[i] = i * tau
    y = np.zeros(partitions + 1)
    y[0] = u0

    for i in range(1, partitions + 1, 1):
        equ = -x+y[i - 1]+tau*f.subs(time, t[i]).subs(u, x)
        y[i] = newton_method(equ)

    return np.array(y)


def explicit_trapezoidal_rule(f, u0=1., partitions=10, a=0., b=1.):
    len = b - a
    tau = len / partitions
    t = np.zeros(partitions + 1)
    for i in range(partitions + 1):
        t[i] = i * tau
    y = np.zeros(partitions + 1)
    y[0] = u0

    for j in range(partitions):
        y[j+1] = y[j]+tau*f.subs(time, t[j]).subs(u, y[j])
        y[j+1] = y[j]+(tau/2)*(f.subs(time, t[j])
            .subs(u, y[j])+f.subs(time, t[j+1]).subs(u, y[j+1]))

    return np.array(y)


def runge_kutta_second_order(f, alpha=1., u0=1., partitions=10, left=0., right=1.):
    len = right-left
    tau = len/partitions
    t = np.zeros(partitions + 1)
    for i in range(partitions + 1):
        t[i] = i * tau
    y = np.zeros(partitions + 1)
    y[0] = u0

    c = np.array([0, alpha])
    b = np.array([1-(1/(2*alpha)), 1/(2*alpha)])
    A = np.array([[0, 0], [alpha, 0]])

    for j in range(partitions):
        k1 = f.subs(time, t[j]).subs(u, y[j])
        k2 = f.subs(time, t[j]+alpha*tau).subs(u, y[j]+tau*k1*alpha)
        y[j+1] = y[j]+(tau/(2*alpha))*((2*alpha-1)*k1+k2)

    return np.array(y)


def adams_bashforth_third_order(f, y, partitions=10, a=0., b=1.):
    len = b - a
    tau = len / partitions
    t = np.zeros(partitions + 1)
    for i in range(partitions + 1):
        t[i] = i * tau

    for j in range(2, partitions, 1):
        y[j+1] = y[j]+(tau/12) * \
                 (23*f.subs(time, t[j]).subs(u, y[j])
                  - 16*f.subs(time, t[j-1]).subs(u, y[j-1])
                  + 5*f.subs(time, t[j-2]).subs(u, y[j-2]))

    return np.array(y)


def runge_kutta_third_order(f, u0=1., partitions=10, a=0., b=1., layers=3):
    len = b - a
    tau = len / partitions
    t = np.zeros(partitions + 1)
    for i in range(partitions + 1):
        t[i] = i * tau
    y = np.zeros(partitions + 1)
    y[0] = u0

    for j in range(layers-1):
        k1 = f.subs(time, t[j]).subs(u, y[j])
        k2 = f.subs(time, t[j]+tau/2).subs(u, y[j]+k1*tau/2)
        k3 = f.subs(time, t[j]+tau).subs(u, y[j]-tau*k1+2*tau*k2)
        y[j+1] = y[j]+(tau/6)*(k1+4*k2+k3)

    return y


f = - (time ** 2 * u ** 2) + (time ** 2 - 0.5) / ((1 + 0.5 * time) ** 2)

euler = euler_method(f)
runge_kutta = runge_kutta_second_order(f)
trapezoidal = explicit_trapezoidal_rule(f)
adams = adams_bashforth_third_order(f, runge_kutta_third_order(f))

print('Explicit Euler Method \n', '\nValues\n', euler, 
'Errors\n', euler - adams)
print('Runge-Kutta Method\n', '\nValues\n', runge_kutta, 
'Errors\n', runge_kutta - adams)
print('Explicit Trapezoidal rule\n', '\nValues\n', trapezoidal, 
'Errors\n', trapezoidal - adams)