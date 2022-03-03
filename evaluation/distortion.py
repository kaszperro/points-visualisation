from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.common import VisualizationAlgorithm, parse_data
from evaluation.utils import EvaluationAlgorithm
from plots import plot_using_importance_matrix


def _close_zero(number, e=1e-6):
    return e if number <= e else number

class Distortion(EvaluationAlgorithm):
    def __init__(self, file_path, closest_element_percentage=1.0):
        self.distances, self.num_elections, self.names_to_indexes = parse_data(
            file_path
        )

        original_file = Path(file_path)
        self.file_name = original_file.stem
        self.closest_element_percentage = closest_element_percentage

        self.save_results_root = Path('evaluation', 'saved', 'distortion', 'results', f'{self.file_name}')

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

        distortion_matrix = np.ones_like(self.distances)

        to_take = int(self.closest_element_percentage * (n-1))

        for i in range(n):
            sorted_distances = self.distances[i].flatten()
            sorted_distances.sort()
            max_distance = sorted_distances[to_take]
            for j in range(n):
                d1 = _close_zero(self.distances[i, j])
                if d1 <= max_distance:
                    d2 = _close_zero(calculated_distances[i, j])
                    if d1 > d2:
                        distortion = d1 / d2
                    else:
                        distortion = d2 / d1

                    distortion_matrix[i, j] = distortion

        mean_distortion = np.mean(distortion_matrix, axis=1)
        print(np.mean(distortion_matrix))

        point_name_to_distortion = {
            k: mean_distortion[v] for k, v in self.names_to_indexes.items()
        }

        return point_name_to_distortion


point_name_to_d = Distortion('../data/positionwise/emd-positionwise-paths-big.csv', 0.1).calculate_from_saved_path(
    'saved/stability/results/emd-positionwise-paths-big-fixed-paths-3runs')

plot_using_importance_matrix('saved/stability/results/emd-positionwise-paths-big-fixed-paths-3runs/emd-positionwise-paths-big_0.csv', point_name_to_d)


#MEAN DIST (10%):
#emd-positionwise-paths-big_0-fixed-4-2runs.csv : 1.0498735726898416
#emd-positionwise-paths-big-fixed-4-3runs/emd-positionwise-paths-big_0.csv :  1.054683614829009
#emd-positionwise-paths-big-fixed-paths-2runs/emd-positionwise-paths-big_0: 1.0523417299172935
#emd-positionwise-paths-big-fixed-paths-3runs/emd-positionwise-paths-big_0: 1.0525321459154457
