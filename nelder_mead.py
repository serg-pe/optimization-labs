import random
from typing import Callable
from math import inf, sqrt

import numpy as np
from numpy.lib.recfunctions import repack_fields

from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt


def f(x: float, y: float) -> float:
    """Функция.

    Args:
        x (float): переменная x.
        y (float): переменная y.

    Returns:
        float: Если точка лежит вне x = [1, 3] и y = [1, 4], то inf, иначе z = f(x, y).
    """
    if x <= 3 and x >= 1 and y <= 4 and y >= 1:
        z = x**2 + x * y - 6 * x - 2 * y + 2
    else:
        z = inf

    return z


def generate_simplex(start_point: np.ndarray, length: float) -> np.ndarray:
    """Создание регулярного симплекса размером (n + 1, n).

    Args:
        start_point (np.ndarray): начальная точка.

    Returns:
        np.ndarray: массив вершин симплекса (строка - координаты вершины).
    """
    simplex = [start_point]
    n = len(start_point)
    r1 = length * (sqrt(n + 1) + n - 1) / (n * sqrt(2))
    r2 = length * (sqrt(n + 1) - 1) / (n * sqrt(2))

    simplex = np.zeros((n, n + 1))
    for row_index, _ in enumerate(simplex):
        for col_index, _ in enumerate(simplex[row_index]):
            if col_index == 0:
                simplex[row_index][col_index] = start_point[row_index]
            elif row_index + 1 == col_index:
                simplex[row_index][col_index] = start_point[row_index] + r1
            else:
                simplex[row_index][col_index] = start_point[row_index] + r2

    simplex = np.transpose(simplex)
    return simplex


def sort_simplex(simplex: np.ndarray, f: Callable) -> np.ndarray:
    """Сортировка симплекса вепшин симплекса по убыванию функции.

    Args:
        simplex (np.ndarray): симплекс.
        f (Callable): заданная функция.

    Returns:
        np.ndarray: отсортированный симплекс.
    """
    sorted_simplex = dict(zip([f(*point) for point in simplex], simplex))
    sorted_simplex = np.array([point[1] for point in sorted(sorted_simplex.items(), reverse=True)])
    return sorted_simplex


def accuracy_achived(simplex: np.ndarray, epsilon: float, f: Callable) -> bool:
    """Проверка условия завершения по достижении заданныой точности

    Args:
        simplex (np.ndarray): симплекс.
        epsilon (float): точность.
        f (Callable): заданная функция.

    Returns:
        [bool]: True, если точность достигнута.
    """
    f_xs = [abs(f(*point) - f(*simplex[-1])) for point in simplex[0:simplex.shape[0] - 1]]
    return max(f_xs) <= epsilon


def recover_simplex(simplex: np.ndarray) -> np.ndarray:
    """Восстановление симплекса.

    Args:
        simplex (np.ndarray): симплекс.

    Returns:
        np.ndarray: симплекс с длиной ребра равной длине ребра начального симплекса между точкой с минимальным значением и следующим за ним.
    """
    edge_length = euclidean(simplex[-1], simplex[-2])
    simplex = generate_simplex(simplex[-1], edge_length)
    return simplex
    
        
        
start_point = [1., 1.]
X = [1., 1.]
l = random.randrange(1, 10) / 10.
r = 0
epsilon = 1.0e-10

max_iter = 1000
recover_simplex_iterations = 50
extremum = None

simplex = generate_simplex(start_point, l)

simplexes = []
for iteration_index in range(max_iter):
    #================
    simplexes.append(simplex)
    #================

    simplex = sort_simplex(simplex, f)
    centroid = np.sum(simplex[1:], axis=0) / (simplex.shape[0] - 1) # центральная точка между центром тяжести
    reflected = 2. * centroid - simplex[0] # отражение

    x_worst = simplex[0].copy()
    simplex[0] = reflected

    if accuracy_achived(simplex, epsilon, f):
        extremum = simplex[-1]
        break

    if f(*simplex[0]) <= f(*x_worst):
        if f(*simplex[0] <= f(*simplex[-1])):
            stretched = centroid + 2. * (simplex[0] - centroid) # растяжение
            if f(*stretched) <= f(*simplex[-1]):
                simplex[0] = stretched
                r += 2
            else:
                r += 3
    else:
        squeezed = centroid + 0.5 * (simplex[0] - centroid) # сжатие
        if f(*squeezed) <= f(*simplex[0]):
            simplex[0] = squeezed
            r += 2
        else:
            simplex[0:len(simplex)] = simplex[0:len(simplex)] + 0.5 * (simplex[0:len(simplex)] - simplex[-1]) # редукция
            r += 1

    if iteration_index % recover_simplex_iterations == 0:
        simplex = recover_simplex(simplex)

    if iteration_index == max_iter - 1:
        extremum = simplex[-1]
        print("Достигнуто максимальное число итераций.")

print(f'\nr = {r}\tМинимум: ({extremum[0]}, {extremum[1]})')
print(f'f(x, y) = {f(*extremum)}')


#================
fig = plt.figure()
axis = fig.gca(projection='3d')

x_axis = np.linspace(1, 3, 100)
y_axis = np.linspace(1, 4, 100)
x_axis, y_axis = np.meshgrid(x_axis, y_axis)
z_axis = np.array([x**2 + x * y - 6 * x - 2 * y + 2 for x, y in zip(x_axis, y_axis)])

axis.plot_surface(x_axis, y_axis, z_axis, shade=True, alpha=0.2, linewidth=3)
for simplex in simplexes:
    simplex = simplex.tolist()
    simplex.append(simplex[0])
    simplex = np.array(simplex)
    axis.plot(xs=simplex[:, 0], ys=simplex[:, 1], zs=[f(x, y) for x, y in zip(simplex[:, 0], simplex[:, 1])])
plt.show()
#================
