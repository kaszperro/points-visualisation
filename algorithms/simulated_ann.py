from pathlib import Path
from random import random

import numpy as np

from algorithms.common import VisualizationAlgorithm, circle_points, _calc_k_with_special_value

from algorithms.kk_energy import get_total_energy as kk_tot_energy

import math

from algorithms.simulated_annealing_energy import get_total_energy


def _rotate(center, radius, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    point = center + radius
    cx, cy = center
    px, py = point

    qx = cx + math.cos(angle) * (px - cx) - math.sin(angle) * (py - cy)
    qy = cy + math.sin(angle) * (px - cx) + math.cos(angle) * (py - cy)
    return qx, qy


class SimRunner:
    def __init__(self, initial_positions, distances, temperature, frozen_node_indexes=None, cooling_temp_factor=0.75,
                 num_stages=10, number_of_trials_for_temp=30, cooling_radius_factor=None, radius=None):
        if cooling_radius_factor is None:
            cooling_radius_factor = cooling_temp_factor

        self.cooling_radius_factor = cooling_radius_factor
        self.positions = initial_positions
        self.distances = distances
        self.num_elections = initial_positions.shape[0]
        if frozen_node_indexes is None:
            frozen_node_indexes = []

        self.frozen_node_indexes = frozen_node_indexes
        if radius is None:
            self.radius = np.amax(self.distances)
        else:
            self.radius = radius

        self.temperature = temperature
        self.cooling_temp_factor = cooling_temp_factor
        self.num_stages = num_stages

        self.number_of_trials_for_temp = number_of_trials_for_temp * self.num_elections

    def _get_rand_point_index(self):
        index = np.random.randint(0, self.num_elections)
        while index in self.frozen_node_indexes:
            index = np.random.randint(0, self.num_elections)
        return index

    def move(self, positions):
        index = self._get_rand_point_index()
        center = positions[index]
        new_position = _rotate(center, self.radius, np.random.uniform(0, 2 * math.pi))
        positions[index] = new_position
        return positions

    def run(self):
        energy = get_total_energy(self.positions, self.distances)
        new_positions = self.move(np.copy(self.positions))
        new_energy = get_total_energy(new_positions, self.distances)
        print(f"Initial Energy: {energy}")

        for i in range(self.num_stages):
            for j in range(self.number_of_trials_for_temp):
                accept = (new_energy < energy or random() < np.exp((energy - new_energy) / self.temperature))
                if accept:
                    print(
                        f"Accepting new energy: {new_energy} temp: {self.temperature}. Temerature Iteration: {j}/{self.number_of_trials_for_temp}. Global Iteration: {i}/{self.num_stages}")
                    self.positions = new_positions
                    energy = new_energy

                new_positions = self.move(np.copy(self.positions))
                new_energy = get_total_energy(new_positions, self.distances)
            self.temperature *= self.cooling_temp_factor
            self.radius *= self.cooling_radius_factor

            kk_tot_e = kk_tot_energy(self.positions, _calc_k_with_special_value(self.distances, 1, [0, 1, 2, 3]), self.distances)
            print(f"KK energy: {kk_tot_e}")
        kk_tot_e = kk_tot_energy(self.positions, _calc_k_with_special_value(self.distances, 1, [0, 1, 2, 3]), self.distances)
        print(f"Final energy: {energy}. Final KK energy: {kk_tot_e}")
        return self.positions


class SimulatedAnnealing(VisualizationAlgorithm):
    def __init__(self, file_path, temperature, num_stages=10, number_of_trials_for_temp=30, cooling_temp_factor=0.75,
                 cooling_radius_factor=None):
        super().__init__(file_path)
        self.num_stages = num_stages
        self.number_of_trials_for_temp = number_of_trials_for_temp
        self.temperature = temperature
        self.cooling_temp_factor = cooling_temp_factor
        self.cooling_radius_factor = cooling_radius_factor

    def _sim_anneal(self, distances, num_elections):
        identity_uniformity_distance = distances[0, 1]
        positions = circle_points(
            [identity_uniformity_distance / 2, identity_uniformity_distance * 2],
            [num_elections // 2, num_elections - num_elections // 2]
        )
        positions[0] = [15.939003137638984, -3.4569181278827243]  # ID
        positions[1] = [-15.939003137638982, 3.456918127882725]  # UN
        positions[2] = [-3.0703888159475805, 13.038665587497677]  # AN
        positions[3] = [3.070388815947584, -13.03866558749768]  # ST

        ann = SimRunner(
            positions, distances,
            self.temperature,
            num_stages=self.num_stages,
            number_of_trials_for_temp=self.number_of_trials_for_temp,
            frozen_node_indexes=[0, 1, 2, 3],
            cooling_temp_factor=self.cooling_temp_factor,
            cooling_radius_factor=self.cooling_radius_factor
        )

        return ann.run()

    def _get_positions(self, distances, num_elections):
        return self._sim_anneal(distances, num_elections)

    def _get_save_file_name(self):
        return Path('simulated_annealing', f'{self.file_name}.csv')

    @staticmethod
    def _get_special_groups_names():
        return ['ID', 'UN', 'AN', 'ST']
