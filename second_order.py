from math import inf
from random import randint
from typing import Iterable, Callable, List

import numpy as np
from matplotlib import pyplot as plt


def f(x: float, y: float) -> float:
    """Функция.

    Args:
        x (float): переменная x.
        y (float): переменная y.

    Returns:
        float: если точка лежит вне x = [1, 3] и y = [1, 4], то inf, иначе z = f(x, y).
    """
    # if x <= 3 and x >= 1 and y <= 4 and y >= 1:
    #     z = x**2 + x * y - 6. * x - 2. * y + 2.
    # else:
    #     z = inf
    z = x**2 + x * y - 6. * x - 2. * y + 2.
    return z


def f_dx(x: float, y: float) -> float:
    """Частная производная первого порядка по x.

    Args:
        x (float): координата x.
        y (float): координата y.

    Returns:
        float:  если точка лежит вне x = [1, 3] и y = [1, 4], то inf, иначе значение частной производной. 
    """
    # if x <= 3 and x >= 1 and y <= 4 and y >= 1:
    #     z_dx = 2. * x + y - 6.
    # else:
    #     z_dx = inf
    z_dx = 2. * x + y - 6.
    return z_dx


def f_dx2(x: float, y: float):
    # if x <= 3 and x >= 1 and y <= 4 and y >= 1:
    #     z_dx2 = 2.
    # else:
    #     z_dx2 = inf
    z_dx2 = 2.
    return z_dx2


def f_dxdy(x: float, y: float):
    # if x <= 3 and x >= 1 and y <= 4 and y >= 1:
    #     z_dxdy = 1.
    # else:
    #     z_dxdy = inf
    z_dxdy = 1.
    return z_dxdy
    

def f_dy(x: float, y: float) -> float:
    """Частная производная первого порядка по y.

    Args:
        x (float): координата x.
        y (float): координата y.

    Returns:
        float: если точка лежит вне x = [1, 3] и y = [1, 4], то inf, иначе значение частной производной.
    """
    # if x <= 3 and x >= 1 and y <= 4 and y >= 1:
    #     z_dy = x - 2.
    # else:
    #     z_dy = inf
    z_dy = x - 2.
    return z_dy


def f_dy2(x: float, y: float) -> float:
    """Частная производная первого порядка по y.

    Args:
        x (float): координата x.
        y (float): координата y.

    Returns:
        float: если точка лежит вне x = [1, 3] и y = [1, 4], то inf, иначе значение частной производной.
    """
    # if x <= 3 and x >= 1 and y <= 4 and y >= 1:
    #     z_dy2 = 1.
    # else:
    #     z_dy2 = inf
    z_dy2 = 1.
    return z_dy2


def f_dydx(x: float, y: float) -> float:
    """Частная производная первого порядка по y.

    Args:
        x (float): координата x.
        y (float): координата y.

    Returns:
        float: если точка лежит вне x = [1, 3] и y = [1, 4], то inf, иначе значение частной производной.
    """
    # if x <= 3 and x >= 1 and y <= 4 and y >= 1:
    #     z_dydx = 1.
    # else:
    #     z_dydx = inf
    z_dydx = 1.
    return z_dydx


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


def get_hessian(X: Iterable[float], partial_derivatives2: List[Callable]) -> np.ndarray:
    """Матрица Гессе.
    """
    hessian = []
    for derivs_line in partial_derivatives2:
        line = []
        for deriv2 in derivs_line:
            line.append(deriv2(*X))
        hessian.append(line)
    hessian = np.array(hessian)
    return hessian


def accuracy_achived(X: np.ndarray, X_prev: np.ndarray, f: Callable, epsilon_y: float) -> bool:
    return abs(f(*X) - f(*X_prev)) <= epsilon_y


def neutons_optimiztation(start_point: np.ndarray, partial_derivatives: np.ndarray, f: Callable, partial_derivatives_2: List[List[Callable]], stride: float=1., stride_split_factor: float=0.1, epsilon_y: float=1.e-10, max_iterations: int=1000, strides_changes: int=4) -> np.ndarray:
    path = []
    
    r = 0
    X = start_point.copy()
    iteration_index = 0
    for iteration_index in range(max_iterations):
        X_prev = X.copy()
        gradient = grad(X, partial_derivatives)
        hessian = get_hessian(X, partial_derivatives_2)
        deltas = np.linalg.solve(hessian * stride, -gradient)

        for _ in range(strides_changes):
            path.append(X)
            X = X_prev + stride * deltas
            if accuracy_achived(X, X_prev, f, epsilon_y):
                return X, r, path

            if f(*X) < f(*X_prev):
                r += 1
                break
            else:
                stride *= stride_split_factor

    if iteration_index == max_iterations - 1:
        print("Достигнуто максимальное число итераций!")

    return X, r, path
    

start_point = np.array([1., 1.])
stride = 1.
stride_split_factor = randint(1, 10) / 10.
r = 0
epsilon_y = 1.0e-10

partial_derivatives = (f_dx, f_dy,)
partial_derivatives_2 = ((f_dx2, f_dxdy), (f_dydx, f_dy2))

max_iterations = 2

X = start_point.copy()
iteration = 0
is_ended = False

extremum, r, path = neutons_optimiztation(start_point, partial_derivatives, f, partial_derivatives_2, max_iterations=max_iterations)

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
