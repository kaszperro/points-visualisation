import time
from pathlib import Path

import numpy as np

from algorithms.common import VisualizationAlgorithm, _calc_k_with_special_value
from algorithms.kamadaY import _optimize_bb
from algorithms.kk_energy import get_energy_dx, get_energy_dy, get_energy_dx_dx, get_energy_dx_dy, get_energy_dy_dx, \
    get_energy_dy_dy, get_total_energy, get_total_energy_dxy
from algorithms.sgd_for_scipy import adam, rmsprop


def _get_pos_k_l_x_y_for_i(positions, k, l, i):
    my_k = np.delete(k[i, :], i)
    my_l = np.delete(l[i, :], i)
    my_positions = np.delete(positions, i, axis=0)

    my_x = positions[i, 0]
    my_y = positions[i, 1]

    return my_positions, my_k, my_l, my_x, my_y


def _get_delta_energy(positions, k, l, x, y):
    return np.sqrt(get_energy_dx(x, y, k, l, positions) ** 2 + get_energy_dy(x, y, k, l, positions) ** 2)


def _optimize_newton(positions, k, l, i, eps=1e-10):
    positions, k, l, x, y = _get_pos_k_l_x_y_for_i(positions, k, l, i)

    delta = _get_delta_energy(positions, k, l, x, y)
    i = 0
    while delta > eps:
        a1 = get_energy_dx_dx(x, y, k, l, positions)
        b1 = get_energy_dx_dy(x, y, k, l, positions)
        c1 = -get_energy_dx(x, y, k, l, positions)

        a2 = get_energy_dy_dx(x, y, k, l, positions)
        b2 = get_energy_dy_dy(x, y, k, l, positions)
        c2 = -get_energy_dy(x, y, k, l, positions)

        dx, dy = np.linalg.solve([[a1, b1], [a2, b2]], [c1, c2])

        x += dx
        y += dy

        if i > 1e4:
            return (x, y), False

        delta = _get_delta_energy(positions, k, l, x, y)
        i += 1
    return (x, y), True


class KamadaXY(VisualizationAlgorithm):
    def __init__(self, file_path, special_k=10000, fixed_positions_path=None, epsilon=0.00001,
                 max_neighbour_distance=None, optim_method='bb'):
        self.special_k = special_k
        self.epsilon = epsilon
        self.max_neighbour_distance = max_neighbour_distance
        self.optim_method = optim_method

        super().__init__(file_path, fixed_positions_path)

    def _reload(self):
        if self.fixed_positions_indexes is not None:
            self.special_indexes = list(range(len(self.fixed_positions_indexes)))
        else:
            self.special_indexes = None

        self.k = _calc_k_with_special_value(self.distances, self.special_k, self.special_indexes)
        self.l = self.distances
        self.epsilon = self.epsilon

        self.optim_method = self.optim_method
        self.max_neighbour_distance = self.max_neighbour_distance

    def _respect_only_close_neighbours(self, max_distance):
        for i in range(self.num_elections):
            for j in range(self.num_elections):
                if abs(self.distances[i, j]) > max_distance:
                    self.k[i, j] = 0.0

    def _get_positions_bb(self, distances, num_elections, positions=None):
        if positions is None:
            positions = self._initial_place_on_circle()

            # positions[self.names_to_indexes['Identity']] = [15, 0]
            # positions[self.names_to_indexes['Uniformity']] = [-15, 0]
            # positions[self.names_to_indexes['Antagonism']] = [0, 15]
            # positions[self.names_to_indexes['Stratification']] = [0, -15]

        pos_copy = np.copy(positions)

        new_positions = _optimize_bb(
            get_total_energy,
            get_total_energy_dxy,
            args=(self.k, self.l, self.special_indexes),
            x0=pos_copy,
            max_iter=int(1e5),
            init_step_size=1e-3,
            max_iter_without_improvement=500
            # stop_energy_val=220

        )

        return new_positions

    def _get_max_derivative(self, positions, special_positions=None):
        max_derivative = 0, 0
        for i in range(0, self.num_elections):
            if special_positions is not None and i in special_positions:
                continue
            pos, k, l, x, y = _get_pos_k_l_x_y_for_i(positions, self.k, self.l, i)

            my_energy = _get_delta_energy(pos, k, l, x, y), i
            if my_energy > max_derivative:
                max_derivative = my_energy

        return max_derivative

    def _get_positions_kk(self, distances, num_elections, positions=None):
        if positions is None:
            positions = self._initial_place_on_circle()

        max_derivative = self._get_max_derivative(positions, self.special_indexes)
        print(max_derivative)
        prev_i = 0
        while max_derivative[0] > self.epsilon:
            max_der, i = max_derivative
            total_energy = get_total_energy(positions, self.k, self.l)
            print(f'Energy: {total_energy}, max der: {max_der}')
            positions[i], succ = _optimize_newton(positions, self.k, self.l, i)
            if not succ:
                positions[i] += np.random.uniform(-10, 10, size=(2,))
            max_derivative = self._get_max_derivative(positions, [0, 1, 2, 3])
            prev_i = i

        return positions

    def _get_positions_adam(self, distances, num_elections, positions=None):
        if positions is None:
            positions = self._initial_place_on_circle()

        pos_copy = np.copy(positions)
        new_positions = adam(
            get_total_energy,
            get_total_energy_dxy,
            x0=pos_copy,
            args=(self.k, self.l, self.special_indexes),
            learning_rate=1.0,
            maxiter=4000
        )
        return new_positions

    def _get_positions_rmsprop(self, distances, num_elections, positions=None):
        if positions is None:
            positions = self._initial_place_on_circle()

        pos_copy = np.copy(positions)
        new_positions = rmsprop(
            get_total_energy,
            get_total_energy_dxy,
            x0=pos_copy,
            args=(self.k, self.l, self.special_indexes),
            learning_rate=0.1,
            maxiter=10000
        )
        return new_positions

    def _get_positions(self, distances, num_elections):
        optim_method_to_fun = {
            'kk': self._get_positions_kk,
            'bb': self._get_positions_bb,
            'adam': self._get_positions_adam,
            'rmsprop': self._get_positions_rmsprop
        }
        start_time = time.time()
        pos = optim_method_to_fun[self.optim_method](distances, num_elections)

        self.k = _calc_k_with_special_value(self.distances, 1, self.special_indexes)
        print("MIDDLE ENERGY:", get_total_energy(pos, self.k, self.l), "TIME:", time.time() - start_time)

        pos = optim_method_to_fun[self.optim_method](distances, num_elections, positions=pos)

        print("FINAL ENERGY:", get_total_energy(pos, self.k, self.l), "TIME:", time.time() - start_time)

        if self.max_neighbour_distance is not None:
            self._respect_only_close_neighbours(self.max_neighbour_distance)

            pos = optim_method_to_fun[self.optim_method](distances, num_elections, positions=pos)
            print("Last adjustments:", get_total_energy(pos, self.k, self.l), "TIME:", time.time() - start_time)

        return pos

    def _get_save_file_name(self):
        return Path('kamada_xy', f'{self.file_name}-{self.optim_method}.csv')
