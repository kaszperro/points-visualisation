from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.common import VisualizationAlgorithm, parse_data
from evaluation.utils import EvaluationAlgorithm


class Monotonicity(EvaluationAlgorithm):
    def __init__(self, file_path):
        self.distances, self.num_elections, self.names_to_indexes = parse_data(
            file_path
        )

        original_file = Path(file_path)
        self.file_name = original_file.stem

        self.save_results_root = Path('evaluation', 'saved', 'monotonicity', 'results', f'{self.file_name}')

    def preprocess(self, visualisation_algorithm: VisualizationAlgorithm):
        positions = visualisation_algorithm.get_positions()
        positions.save(self.save_results_root, overridden_file_name=f'{self.file_name}.csv')

        return self.save_results_root

    def calculate_from_saved_path(self, root_path):
        root_path = Path(root_path)
        csv_path = list(root_path.glob('*.csv'))[0]
        df = pd.read_csv(csv_path)

        indexes_to_names = {
            v: k for k, v in self.names_to_indexes.items()
        }

        point_names = list(self.names_to_indexes.keys())

        calculated_distances = np.zeros_like(self.distances)
        n = len(point_names)
        for i in range(n):
            print(f"{i}/{n}")
            for j in range(i + 1, n):
                point1_name = indexes_to_names[i]
                point2_name = indexes_to_names[j]

                row1 = df[df['names'] == point1_name].iloc[0]
                row2 = df[df['names'] == point2_name].iloc[0]

                pos1 = np.array([row1['x'], row1['y']])
                pos2 = np.array([row2['x'], row2['y']])
                calculated_distance_p12 = np.linalg.norm(pos1 - pos2)

                calculated_distances[i, j] = calculated_distance_p12
                calculated_distances[j, i] = calculated_distance_p12

        num_wrong = 0
        num_correct = 0

        for i in range(n):
            print(f"{i}/{n}")
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    original_distance_p12 = self.distances[i, j]
                    original_distance_p13 = self.distances[i, k]

                    calculated_distance_p12 = calculated_distances[i, j]
                    calculated_distance_p13 = calculated_distances[i, k]

                    if np.sign(original_distance_p12 - original_distance_p13) != np.sign(
                            calculated_distance_p12 - calculated_distance_p13):
                        num_wrong += 1
                    else:
                        num_correct += 1

        print(num_correct, num_wrong, num_correct / (num_correct + num_wrong))
#16302100 1223496 0.9301880518071968

Monotonicity('../data/positionwise/emd-positionwise-paths-big.csv').calculate_from_saved_path(
    'saved/stability/results/emd-positionwise-paths-big-fixed-4-2runs')
