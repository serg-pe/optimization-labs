from typing import Callable, Iterable
from functools import reduce
import random
from math import inf

import numpy as np

import matplotlib.pyplot as plt


def f(x: float, y: float) -> float:
    if x <= 3 and x >= 1 and y <= 4 and y >= 1:
        z = x**2 + x * y - 6 * x - 2 * y + 2
    else:
        z = inf

    return z


def accuracy_achived(X_prev: np.ndarray, X_cur: np.ndarray, eps_x: float, eps_y: float, f: Callable[..., float]) -> bool:
    x_condition = all([elem <= eps_x for elem in np.abs(X_prev - X_cur)])
    f_x_condition = abs(f(*X_prev) - f(*X_cur)) <= eps_y
    return x_condition or f_x_condition


def extract_minimal(points: Iterable[np.ndarray], f: Callable):
    minimal = points[0]
    for point in points[1:]:
        if f(*point) < f(*minimal):
            minimal = point
    return minimal


X = [1., 2.]
strides = np.array([random.randrange(1, 10) / 10 for _ in range(len(X))])
print(f'Шаги: {strides}')
r = 1
epsilon_x = 1.0e-10
epsilon_y = 1.0e-10
max_iter = 1000

aceleration_factor = 2.

X = np.array(X)
stridex = np.array(strides)

extremum = None

plt_X0 = X.copy()
plt_points = []

for iteration_index in range(max_iter):
    X_prev = X.copy()
    plt_points.append(X)

    strides_local = strides.copy()
    points = []
    for direction_index, _ in enumerate(X):
        L = np.zeros(X.shape)
        L[direction_index] = 1.
        if min(f(*(X + strides * L)), f(*(X - strides * L))) > f(*X):
            strides_local[direction_index] = 0.
        positive_direction = X + L * strides_local
        negative_direction = X - L * strides_local
        points.append(positive_direction)
        points.append(negative_direction)
    X = extract_minimal(points, f)

    if np.all(X == X_prev):
        strides /= 2.
    else:
        if accuracy_achived(X_prev, X, epsilon_x, epsilon_y, f):
            extremum = X
            break
        else:
            X = X_prev + (X - X_prev)
            r += 1

    if iteration_index == max_iter - 1:
        extremum = X
        print("Достигнуто максимальное число итераций!")

print(f'\nr = {r}\tМинимум: ({extremum[0]}, {extremum[1]}), шаги: {strides}')
print(f'f(x, y) = {f(*extremum)}')


plt_points.append(extremum)
x_axis = np.linspace(1, 3, 100)
y_axis = np.linspace(1, 4, 100)
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
z_axis = np.array([x**2 + x * y - 6 * x - 2 * y + 2 for x, y in zip(x_axis, y_axis)])
plt_points = np.array(plt_points)

fig = plt.figure()
axis = fig.gca(projection='3d')

axis.plot_surface(x_axis, y_axis, z_axis)
axis.scatter3D(plt_X0[0], plt_X0[1], f(*plt_X0), s=15, c='green')
axis.plot(xs=plt_points[:, 0:1].reshape(plt_points.shape[0]), ys=plt_points[:, 1:2].reshape(plt_points.shape[0]), zs=[f(*point) for point in plt_points])
axis.scatter3D(extremum[0], extremum[1], f(*extremum), s=10, c='red')

plt.show()
