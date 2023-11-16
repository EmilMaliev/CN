import numpy as np
import math as m


def k(x, s):
    return np.exp(-x - s)

def k1(x, s):
    return np.exp(-x -s) * (s <= x)

def f(x):
    return 0.5 * (np.exp(-x) + np.exp(-3 * x))


def method_of_mechanical_quadratures(k, f, method, equ_type, N=10, l=0.3, a=0., b=1.):
    h = (b - a) / N
    n = N + 1
    coefficients = np.full(shape=n, fill_value=h)
    x = np.arange(a, b + h, h)
    A = np.zeros((n, n))
    K = np.zeros((n, n))

    if method == 'Right Riemann sum':
        coefficients[0] = 0
        
    if method == 'Trapezoidal rule':
        coefficients[0], coefficients[N] = h / 2, h / 2

    if equ_type == 'Fredholm':
        for i in range(n):
            for j in range(n):
                K[i][j] = k(x[i], x[j])

    elif equ_type == 'Volterra':
        for i in range(n):
            for j in range(n):
                if i <= j:
                    K[i][j] = k(x[i], x[j])
                else:
                    K[i][j] = 0

    for i in range(n):
        for j in range(n):
            A[i][j] = 1 - l * coefficients[i] * K[i][j] if i == j \
                else -l * h * K[i][j]
    F = np.zeros(n)
    for i in range(n):
        F[i] = f(x[i])
    y = np.linalg.solve(A, F)

    return y, coefficients


def value_in_point(value, k, f, method, equ_type, N=10, l=0.3, a=0., b=1.):
    h = (b - a) / N
    n = N + 1
    x = np.arange(a, b + h, h)
    y, coefficients = method_of_mechanical_quadratures(k, f, method, equ_type, N, l, a, b)
    summands = np.zeros(n)
    for i in range(n):
        summands[i] = coefficients[i] * y[i] * k(value, x[i])

    return l * sum(summands) + f(value)


############################################
#This is for for Emchik 
############################################


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
    print(
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
    print(
        value_in_point(x[i], k, f, 'Trapezoidal rule', 'Volterra', l=0.3, N=10),
        solution_volterra(x[i], 100, f, 0.3),
        solution_volterra(x[i], 100, f, 0.3) - value_in_point(x[i], k, f, 'Trapezoidal rule', 'Volterra', l=0.3, N=10)
        )
print('Fredholm in 1/2.2')
print   (
        value_in_point(1/2.2, k, f, 'Trapezoidal rule', 'Volterra', l=0.3, N=10),
        solution_fredholm(1/2.2, 100, f, 0.3),
        solution_fredholm(1/2.2, 100, f, 0.3) - value_in_point(1/2.2, k, f, 'Trapezoidal rule', 'Volterra', l=0.3, N=10)
        )

