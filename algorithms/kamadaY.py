from pathlib import Path

import numpy as np
from scipy import optimize

from algorithms.common import VisualizationAlgorithm, _place_on_circle_2d, _close_zero, _upper_tri_sum


def _calc_total_energy(positions, k, l):
    positions_delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    pos_squared = positions_delta[:, :, 0] ** 2 + positions_delta[:, :, 1] ** 2

    my_matrix = k * (pos_squared + l ** 2 - 2 * l * np.sqrt(pos_squared)) / 2

    return _upper_tri_sum(my_matrix)


def _energy_all(new_ys, positions, k, l):
    positions[:, 1] = new_ys
    return _calc_total_energy(positions, k, l)


def _energy_dy_all(new_ys, positions, k, l):
    positions[:, 1] = new_ys
    positions_delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    pos_squared = positions_delta[:, :, 0] ** 2 + positions_delta[:, :, 1] ** 2
    np.fill_diagonal(pos_squared, 1)
    pos_squared = _close_zero(pos_squared)
    matrix = k * (positions_delta[:, :, 1] - l * positions_delta[:, :, 1] / np.sqrt(pos_squared))
    np.fill_diagonal(matrix, 0)
    matrix_sum = matrix.sum(axis=1)

    return matrix_sum


def _energy_dy(y, x, k, l, positions):
    """
    we calc derivative energy for i

    :param y: y pos
    :param x:  x pos
    :param k: np.delete(k[i, :], i)
    :param l: np.delete(l[i, :], i)
    :param positions: all positions
    :return:
    """
    xs_delta = x - positions[:, 0]
    ys_delta = y - positions[:, 1]

    my_vector = k * (ys_delta - l * ys_delta / np.sqrt(_close_zero(xs_delta ** 2 + ys_delta ** 2)))

    return my_vector.sum()


def _energy_dy_dy(y, x, k, l, positions):
    """
    calc derivative derivative energy for i

    :param k:
    :param l:
    :param positions:
    :param i:
    :return:
    """
    xs_delta_sq = (x - positions[:, 0]) ** 2
    ys_delta_sq = (y - positions[:, 1]) ** 2

    my_vector = k * (1 - l * xs_delta_sq / np.power(_close_zero(xs_delta_sq + ys_delta_sq), 3 / 2))
    return my_vector.sum()


def _energy_dy_dy_dy(y, x, k, l, positions):
    xs_delta = x - positions[:, 0]
    ys_delta = y - positions[:, 1]

    xs_sq = xs_delta ** 2
    ys_sq = ys_delta ** 2

    my_vector = 3 * k * l * xs_sq * ys_delta / np.power(xs_sq + ys_sq, 5 / 2)
    return my_vector.sum()


def sgd_bb(grad, init_step_size, d, max_epoch=100, args=None, m=None, x0=None, beta=None, phi=lambda k: k,
           func=None, verbose=True):
    """
        SGD with Barzilai-Borwein step size for solving finite-sum problems
        grad: gradient function in the form of grad(x, idx), where idx is a list of induces
        init_step_size: initial step size
        n, d: size of the problem
        m: step sie updating frequency
        beta: the averaging parameter
        phi: the smoothing function in the form of phi(k)
        func: the full function, f(x) returning the function value at x
    """
    if not isinstance(m, int) or m <= 0:
        m = d
        if verbose:
            print('Info: set m=n by default')

    if not isinstance(beta, float) or beta <= 0 or beta >= 1:
        beta = 10 / m
        if verbose:
            print('Info: set beta=10/m by default')

    if x0 is None:
        x = np.zeros(d)
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    step_size = init_step_size
    c = 1
    for k in range(max_epoch):
        x_tilde = x.copy()
        # estimate step size by BB method
        if k > 1:
            s = x_tilde - last_x_tilde
            y = grad_hat - last_grad_hat
            step_size = np.linalg.norm(s) ** 2 / abs(np.dot(s, y)) / m
            # smoothing the step sizes
            if phi is not None:
                c = c ** ((k - 2) / (k - 1)) * (step_size * phi(k)) ** (1 / (k - 1))
                step_size = c / phi(k)

        if verbose:
            full_grad = grad(x, *args)
            output = 'Epoch.: %d, Step size: %.2e, Grad. norm: %.2e' % \
                     (k, step_size, np.linalg.norm(full_grad))
            if func is not None:
                output += ', Func. value: %e' % func(x, *args)
            print(output)

        if k > 0:
            last_grad_hat = grad_hat
            last_x_tilde = x_tilde
        if k == 0:
            grad_hat = np.zeros(d)

        g = grad(x, *args)
        x -= step_size * g
        # average the gradients
        grad_hat = beta * g + (1 - beta) * grad_hat

    return x


def _optimize_bb(func, grad_func, args, x0, max_iter, init_step_size, stop_energy_val=None, max_iter_without_improvement=8000):
    if isinstance(init_step_size, float):
        init_step_size = [init_step_size, init_step_size]

    init_step_size = np.asarray(init_step_size)
    is_2d = len(x0.shape) == 2

    prev_x = x0.copy()
    x = x0.copy()
    prev_grad = grad_func(prev_x, *args)

    min_energy = 1e15
    min_energy_snap = x0.copy()
    min_energy_iter = 0


    for i in range(max_iter):
        current_energy = func(x, *args)
        if current_energy < min_energy:
            min_energy = current_energy
            min_energy_snap = x.copy()
            min_energy_iter = i
        elif i - min_energy_iter > max_iter_without_improvement:
            return min_energy_snap


        print(f'Energy: {current_energy}: {min_energy}, grad norm: {np.linalg.norm(prev_grad)} {i}')
        if stop_energy_val is not None and current_energy < stop_energy_val:
            return min_energy_snap
        s = x - prev_x
        g = grad_func(x, *args)
        y = g - prev_grad

        if i > 0:
            denominator = abs(np.tensordot(s, y, [0, 0]))
            if is_2d:
                denominator = denominator.diagonal()
            step_size = np.linalg.norm(s, axis=0) ** 2 / denominator
        else:
            step_size = init_step_size

        prev_grad = g
        prev_x = x
        #
        # if step_size[0] < 0.003:
        #     step_size[0] *= 3
        #     step_size[1] *= 3
        x = x - step_size * g



    return min_energy_snap


class KamadaY(VisualizationAlgorithm):

    def __init__(self, file_path, k=1, epsilon=0.3, max_neighbour_distance=None):
        super().__init__(file_path)

        square_dist = self.distances ** 2
        np.fill_diagonal(square_dist, 1)
        _close_zero(square_dist)
        self.k = k / square_dist
        np.fill_diagonal(self.k, 0)
        self.k[0] = 0
        self.k[1] = 0
        self.l = self.distances
        self.epsilon = epsilon

        if max_neighbour_distance is not None:
            self._respect_only_close_neighbours(max_neighbour_distance)

    def _respect_only_close_neighbours(self, max_x_distance):
        for i in range(self.num_elections):
            x_i = self.distances[0, i]
            for j in range(self.num_elections):
                x_j = self.distances[0, j]
                if abs(x_i - x_j) > max_x_distance:
                    self.k[i, j] = 0.0

    def _get_k_l_pos_x_y_for_i(self, positions, i):
        my_k = np.delete(self.k[i, :], i)
        my_l = np.delete(self.l[i, :], i)
        my_positions = np.delete(positions, i, axis=0)

        my_x = positions[i, 0]
        my_y = positions[i, 1]

        return my_k, my_l, my_positions, my_x, my_y

    def _get_max_derivative(self, positions):
        max_derivative = 0, 0
        for i in range(2, self.num_elections):
            k, l, pos, x, y = self._get_k_l_pos_x_y_for_i(positions, i)

            my_energy = _energy_dy(y, x, k, l, pos) ** 2, i
            if my_energy > max_derivative:
                max_derivative = my_energy

        return max_derivative

    def _get_positions_bb(self, init_positions):
        # positions = self._place_on_circle()
        positions = init_positions

        pos_copy = np.copy(positions)

        new_ys = _optimize_bb(
            _energy_all,
            _energy_dy_all,
            args=(pos_copy, self.k, self.l),
            x0=positions[:, 1],
            max_iter=int(1e5),
            init_step_size=1e-3,
            # stop_energy_val=220

        )

        positions[:, 1] = new_ys
        return positions

    def _get_positions_kk(self, distances, num_elections):
        positions = _place_on_circle_2d(distances, num_elections)
        # positions = self.positions.copy()

        max_derivative = self._get_max_derivative(positions)
        print(max_derivative)
        while max_derivative[0] > self.epsilon:
            max_der, i = max_derivative
            print(f'Energy: {_calc_total_energy(positions, self.k, self.l)}, max der: {max_der}')
            k, l, pos, x, y = self._get_k_l_pos_x_y_for_i(positions, i)

            try:
                positions[i, 1] = optimize.newton(
                    _energy_dy,
                    y,
                    fprime=_energy_dy_dy,
                    fprime2=_energy_dy_dy_dy,
                    args=(x, k, l, pos),
                    maxiter=int(1e4),
                )

            except RuntimeError as e:
                positions[i, 1] += np.random.uniform(-10, 10)

            max_derivative = self._get_max_derivative(positions)

        return positions

    def _get_positions(self, distances, num_elections):
        positions = self._get_positions_kk(distances, num_elections)
        return self._get_positions_bb(positions)

    def _get_save_file_name(self):
        return Path('kamada_y', f'{self.file_name}-kk-bb-top.csv')
