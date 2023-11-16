# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 22:38:08 2022

@author: Professional
"""
import math
import sympy as sym
import numpy as np
from sympy import symbols


time, u, x = symbols('time u x')


def newton_method(f, x0=0., eps=10**-15):
    x1 = x0
    x2 = x0+1.0
    df = sym.diff(f, x)

    while np.abs(x1 - x2) >= eps:
        x2 = x1
        x1 = x1 - (f.subs(x, x1) / df.subs(x, x1))

    return x1


def euler_method(f, u0=1., partitions=10, a=1., b=2.):
    len = b-a
    tau = len / partitions
    print(tau)
    t = np.zeros(partitions + 1)
    for i in range(partitions + 1):
        t[i] = i * tau + a
    y = np.zeros(partitions + 1)
    y[0] = u0

    for i in range(1, partitions + 1, 1):
        equ = -x+y[i - 1]+tau*f.subs(time, t[i]).subs(u, x)
        y[i] = newton_method(equ)

    return np.array(y)

def mpppt2(f, u0=1., partitions=10, a=1., b=2.):
    len = b - a
    tau = len / partitions
    t = np.zeros(partitions + 1)
    for i in range(partitions + 1):
        t[i] = i * tau + a
    y = np.zeros(partitions + 1)
    y[0] = u0

    for j in range(partitions):
        tmp = y[j] + tau* 0.5 * f.subs(time, t[j]).subs(u, y[j])
        y[j + 1] = y[j] + f.subs(u, tmp).subs(time, t[j] + 0.5 * tau) * tau

    return np.array(y)

def mpppt3(f, u0=1., partitions=10, a=1., b=2.):
    len = b - a
    tau = len / partitions
    t = np.zeros(partitions + 1)
    for i in range(partitions + 1):
        t[i] = i * tau + a
    y = np.zeros(partitions + 1)
    y[0] = u0

    for j in range(partitions):
        tmp1 = y[j] + tau* (1/3) * f.subs(time, t[j]).subs(u, y[j])
        tmp2 = y[j] + tau* (2/3) * f.subs(time, t[j] + (1/3) * tau ).subs(u, tmp1)
        y[j + 1] = y[j] + tau * 0.25 * (f.subs(time, t[j]).subs(u, y[j]) + 3* f.subs(time, t[j] + (2/3) * tau).subs(u, tmp2))

    return np.array(y)


def adams_bashforth_third_order(f, y, partitions=10, a=1., b=2.):
    len = b - a
    tau = len / partitions
    t = np.zeros(partitions + 1)
    for i in range(partitions + 1):
        t[i] = i * tau + a

    for j in range(2, partitions, 1):
        y[j+1] = y[j]+(tau/12) * \
                 (23*f.subs(time, t[j]).subs(u, y[j])
                  - 16*f.subs(time, t[j-1]).subs(u, y[j-1])
                  + 5*f.subs(time, t[j-2]).subs(u, y[j-2]))

    return np.array(y)


def runge_kutta_second_order(f, alpha=1., u0=1., partitions=10, left=1., right=2.):
    len = right-left
    tau = len/partitions
    t = np.zeros(partitions + 1)
    for i in range(partitions + 1):
        t[i] = i * tau + left
    y = np.zeros(partitions + 1)
    y[0] = u0

    for j in range(partitions):
        k1 = f.subs(time, t[j]).subs(u, y[j])
        k2 = f.subs(time, t[j]+alpha*tau).subs(u, y[j]+tau*k1*alpha)
        y[j+1] = y[j]+(tau/(2*alpha))*((2*alpha-1)*k1+k2)

    return np.array(y)

f = ((u ** 2) * sym.log(time) - u) / time

euler = euler_method(f)
runge_kutta = runge_kutta_second_order(f)
mpppt = mpppt2(f)
adams = adams_bashforth_third_order(f, mpppt3(f))

print('Explicit Euler Method \n', '\nValues\n', euler, 
'Errors\n', euler - adams)
print('Runge-Kutta Method\n', '\nValues\n', runge_kutta, 
'Errors\n', runge_kutta - adams)
print('Explicit Mpppt2\n', '\nValues\n', mpppt, 
'Errors\n', mpppt - adams)
print('Adams Method \n', '\nValues\n', adams)