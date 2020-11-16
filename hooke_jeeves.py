from typing import Callable
import random
from math import inf

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(x: float, y: float) -> float:
    if x <= 3 and x >= 1 and y <= 4 and y >= 1:
        z = x**2 + x * y - 6 * x - 2 * y + 2
    else:
        z = inf

    return z


def min_next_point(X: np.ndarray, strides: np.ndarray, f: Callable[..., float]) -> np.array:
    for i, _ in enumerate(X):
        e = np.zeros(np.shape(X))
        e[i] = 1.
        if f(*(X + strides * e)) < f(*X):
            X = X + strides * e
        elif f(*(X - strides * e)) < f(*X):
            X = X - strides * e

    return X


def accuracy_achived(X_prev: np.ndarray, X_cur: np.ndarray, eps_x: float, eps_y: float, f: Callable[..., float]) -> bool:
    x_condition = all([elem <= eps_x for elem in np.abs(X_prev - X_cur)])
    f_x_condition = abs(f(*X_prev) - f(*X_cur)) <= eps_y
    return x_condition or f_x_condition


X = [1., 1.]
strides = np.array([random.randrange(1, 10) / 10 for _ in range(len(X))])
print(f'Шаги: {strides}')
r = 1
epsilon_x = 1.0e-10
epsilon_y = 1.0e-10
max_iter = 100000

X = np.array(X)
stridex = np.array(strides)

plt_X0 = X
plt_points = []
for _ in range(max_iter):
    X_prev = X

    while  f(*X_prev) <= f(*X):
        X = X_prev
        X = min_next_point(X, strides, f)
    
    if accuracy_achived(X, X_prev, epsilon_x, epsilon_y, f):
        break
    plt_points.append([*X, f(*X)])

    strides = strides / 2.
    r += 1

print(f'\nr = {r}\tМинимум: ({X[0]}, {X[1]})')
print(f'f(x, y) = {f(*X)}')

x_axis = np.linspace(1, 3, 100)
y_axis = np.linspace(1, 4, 100)
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
z_axis = np.array([x**2 + x * y - 6 * x - 2 * y + 2 for x, y in zip(x_axis, y_axis)])
plt_points = np.array(plt_points)

fig = plt.figure()
axis = fig.gca(projection='3d')

axis.plot_surface(x_axis, y_axis, z_axis)
axis.scatter3D(plt_X0[0], plt_X0[1], f(*plt_X0), s=15, c='green')
axis.scatter3D(plt_points[:, 0:1], plt_points[:, 1:2], plt_points[:, 2:3], s=10, c='red')

plt.show()
