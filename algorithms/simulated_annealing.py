from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from simanneal import Annealer
import random

from algorithms.common import parse_data, save_positions


class PositionsAnnealProblem(Annealer):
    copy_strategy = 'method'

    def __init__(self, state, distances):
        super().__init__(state)
        self.init_positions = state.copy()
        self.distances = distances

        x_space = self.distances[0, 1]

        self.lower_bound = np.zeros_like(self.init_positions)
        self.upper_bound = np.zeros_like(self.init_positions)

        self.lower_bound[:, 0] = self.init_positions[:, 0]  # - x_space * 0.01
        self.upper_bound[:, 0] = self.init_positions[:, 0]  # + x_space * 0.01

        self.lower_bound[:, 1] = self.init_positions[:, 1] - x_space * 1.0
        self.upper_bound[:, 1] = self.init_positions[:, 1] + x_space * 1.0

        num_elections = self.state.shape[0]

        # self.x_diff = np.expand_dims(self.init_positions[:, 0], 1).repeat(num_elections, axis=1)
        # self.x_diff = np.abs(self.x_diff - self.x_diff.T)
        # self.x_diff = self.x_diff ** 2
        # self.x_diff = self.x_diff * self.distances
        self.x_diff = self.distances ** 2

        self.x_diff[self.x_diff == 0] = 0.000001

    def _calc_energy_for_i(self, i):
        my_distances = np.linalg.norm(self.state[i] - self.state, axis=1)
        errors = np.abs(my_distances - self.distances[i, :])  # / self.x_diff[i, :]
        return np.sum(errors)

    def move(self):
        num_elections = self.state.shape[0]
        i = random.randint(2, num_elections - 1)

        start_e = self._calc_energy_for_i(i)

        # self.state[i, 0] += np.random.uniform(-10, 10)
        self.state[i, 1] += np.random.uniform(-50, 50)
        self.state[i] = np.clip(self.state[i], self.lower_bound[i], self.upper_bound[i])

        return self._calc_energy_for_i(i) - start_e

    def energy(self):
        num_elections = self.state.shape[0]
        my_distances = np.expand_dims(self.state, 2).repeat(num_elections, axis=2)
        my_distances = np.linalg.norm(my_distances - my_distances.T, axis=1)

        # mozna wartosc wzgledna
        errors = np.abs(my_distances - self.distances)  # / self.x_diff
        return np.sum(errors)


class Bordawise:
    def __init__(self, file_path):
        self.distances, self.num_elections, self.indexes_to_names = parse_data(file_path)
        self.file_name = Path(file_path).stem
        self.positions = self.get_positions()

    def get_positions(self):
        positions = np.zeros((self.num_elections, 2))

        for i in range(self.num_elections):
            positions[i, 0] = self.distances[i, 0]
            if i > 1:
                positions[i, 1] = np.random.uniform(-50, 50)

        pap = PositionsAnnealProblem(positions, self.distances)
        # auto_schedule = pap.auto(minutes=0.3)
        # print(auto_schedule)

        pap.set_schedule({
            'tmax': 310000.0,
            'tmin': 0.000000001,
            'steps': 9500000,
            'updates': 500
        })
        # self.positions=positions
        # print(self.positions.shape)
        # pap.set_schedule(auto_schedule)
        # self.plot()

        state, e = pap.anneal()
        print(e)

        return state

    #
    def plot(self, save_path=None):
        plt.scatter(self.positions[:, 0], self.positions[:, 1])
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=300)

        plt.close()

    def save(self, root_path='saved_results'):
        p = Path(root_path, 'network', f'{self.file_name}.csv')
        save_positions(p, self.positions, self.indexes_to_names)

        # self.plot(Path(p, 'plot.png'))

#  {'tmax': 24000000.0, 'tmin': 2e-31, 'steps': 3400, 'updates': 100}
#  12299807.5133376752299807.51
#  30 min
