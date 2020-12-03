from functools import partial
from out_of_domain_exception import OutOfDomainException
import random
from math import inf, sqrt
from typing import Callable, List, Iterable

import numpy as np

import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient
from numpy.lib.shape_base import split


def f(x: float, y: float) -> float:
    """Функция.

    Args:
        x (float): переменная x.
        y (float): переменная y.

    Returns:
        float: если точка лежит вне x = [1, 3] и y = [1, 4], то inf, иначе z = f(x, y).
    """
    if x <= 3 and x >= 1 and y <= 4 and y >= 1:
        z = x**2 + x * y - 6 * x - 2 * y + 2
    else:
        z = inf
    return z


def f_dx(x: float, y: float) -> float:
    """Частная производная первого порядка по x.

    Args:
        x (float): координата x.
        y (float): координата y.

    Returns:
        float:  если точка лежит вне x = [1, 3] и y = [1, 4], то inf, иначе значение частной производной. 
    """
    if x <= 3 and x >= 1 and y <= 4 and y >= 1:
        z_dx = 2 * x + y - 6
    else:
        z_dx = inf
    return z_dx

def f_dy(x: float, y: float) -> float:
    """Частная производная первого порядка по y.

    Args:
        x (float): координата x.
        y (float): координата y.

    Returns:
        float: если точка лежит вне x = [1, 3] и y = [1, 4], то inf, иначе значение частной производной.
    """
    if x <= 3 and x >= 1 and y <= 4 and y >= 1:
        z_dy = x - 2
    else:
        z_dy = inf
    return z_dy


def grad(X: Iterable[float], derivs: Iterable[Callable]):
    """Градиент функции в точке.

    Args:
        X (Iterable[float]): координаты точки.
        derivs (Iterable[Callable]): частные производные по всем параметрам функции.

    Returns:
        [type]: градиент функции в точке.
    """
    gradient = [deriv(*X) for deriv in derivs]
    gradient = np.array(gradient)
    return gradient


def euclidean_norm(vector: np.ndarray) -> float:
    """Евклидова норма (длина вектора).

    Args:
        vector (np.ndarray): вектор.

    Returns:
        float: длина вектора.
    """
    norm = sqrt(np.sum(np.dot(vector, vector)))
    return norm


def accuracy_achived(grad: np.ndarray, epsilon_grad: float) -> bool:
    return euclidean_norm(gradient) <= epsilon_grad

    

path = []

start_point = np.array([1., 1.])
r = 0
stride = 1.
split_factor = random.randint(1, 10) / 10.

epsilon_grad = 1.0e-10

max_iter = 10000

partial_derivatives = (f_dx, f_dy,)

X, X_prev = start_point.copy(), start_point.copy()
for iteration in range(max_iter):
    try:

        gradient = grad(X, partial_derivatives)
        # если произошёл выход за область определения функции, то уменьшение шага и восстановление X
        if any(grad_direction == inf for grad_direction in gradient):
            raise OutOfDomainException()
        # ------------
        X_prev = X.copy()
        path.append(X)
        S = -gradient/euclidean_norm(gradient)
        X = X_prev + stride * S

        if f(*X_prev) - f(*X) >= 0.5 * stride * euclidean_norm(gradient):
            stride = stride * split_factor
        else:
            if accuracy_achived(gradient, epsilon_grad):
                break
            else:
                r += 1

    except OutOfDomainException:
        X = X_prev.copy()
        stride = stride * split_factor
    
    finally:
        if iteration == max_iter - 1:
                print("Достигнуто максимальное число итераций!")

extremum = X
path.append(extremum)

print(f'r = {r}\tМинимум: ({extremum[0]}, {extremum[1]})')
print(f'f(x, y) = {f(*extremum)}')

fig = plt.figure()
axis = fig.gca(projection='3d')

x_axis = np.linspace(1, 3, 100)
y_axis = np.linspace(1, 4, 100)
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
z_axis = np.array([x**2 + x * y - 6 * x - 2 * y + 2 for x, y in zip(x_axis, y_axis)])

axis.plot_surface(x_axis, y_axis, z_axis, shade=True, alpha=0.2, linewidth=3)

path = np.array(path)
axis.scatter(start_point[0], start_point[1], f(*start_point), c='green')
reshape_len = path.shape[0]
axis.plot(xs=path[:, 0:1].reshape(reshape_len), ys=path[:, 1:2].reshape(reshape_len), zs=[f(*path_point) for path_point in path])
axis.scatter3D(extremum[0], extremum[1], f(*extremum), s=10, c='red')

plt.show()
