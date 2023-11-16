import numpy as np


def k(x, s):
    return np.exp(-x - s)

def k1(x, s):
    return np.exp(-x - s) * (s <= x)

def f(x):
    return 0.5 * (np.exp(-x) + np.exp(-3 * x))


def method_of_mechanical_quadratures(k, f, N=10, l=1., a=0., b=1.):
    h = (b - a) / N
    n = N + 1
    coefficients = np.full(shape=n, fill_value=h)
    coefficients[0] = 0
    x = np.arange(a, b + h, h)
    A = np.zeros((n, n))
    K = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            K[i][j] = k(x[i], x[j])

    for i in range(n):
        for j in range(n):
            A[i][j] = 1 - l * coefficients[i] * K[i][j] if i == j \
                else -l * h * K[i][j]
                
    F = np.zeros(n)
    for i in range(n):
        F[i] = f(x[i])
    y = np.linalg.solve(A, F)

    return y, coefficients


def value_in_point(value, k, f,  N=10, l=1., a=0., b=1.):
    h = (b - a) / N
    n = N + 1
    x = np.arange(a, b + h, h)
    y, coefficients = \
        method_of_mechanical_quadratures(k, f, N, l, a, b)
    summands = np.zeros(n)
    for i in range(n):
        summands[i] = coefficients[i] * y[i] * k(value, x[i])

    return l * sum(summands) + f(value)


print(value_in_point(1/2.2, k, f, l=0.3))
points = np.arange(0, 1., 0.1)
for i in range(10):
    print(value_in_point(points[i], k, f, l=0.3))