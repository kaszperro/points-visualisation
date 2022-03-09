from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.common import VisualizationAlgorithm, parse_data
from evaluation.utils import EvaluationAlgorithm
from plots import plot_using_importance_matrix, plot_using_map


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
        return positions.save(self.save_results_root, overridden_file_name=f'{self.file_name}.csv')

    def calculate_from_saved_path(self, csv_path):
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

        num_wrong_matrix = [0 for _ in range(n)]

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
                        num_wrong_matrix[i] += 1
                        num_wrong_matrix[j] += 1
                        num_wrong_matrix[k] += 1
                    else:
                        num_correct += 1

        print(num_correct, num_wrong, num_correct / (num_correct + num_wrong))

        point_name_to_monotonicity = {
            k: num_wrong_matrix[v] for k, v in self.names_to_indexes.items()
        }

        return point_name_to_monotonicity
#16302100 1223496 0.9301880518071968

for i in range(1):
    csv_path = Path(f'saved/stability/results/emd-positionwise-paths-big/emd-positionwise-paths-big_{i}.csv')
    print(csv_path.stem)
    point_name_to_m = Monotonicity('../data/positionwise/emd-positionwise-paths-big.csv').calculate_from_saved_path(
        csv_path)

    plot_using_importance_matrix(csv_path, point_name_to_m, show=False, save_path=Path(csv_path.parent, f'{csv_path.stem}_mono.png'))
    plot_using_map(csv_path, '../map.csv', False)

# RESULTS - all distances:
# KAMADA:
# emd-positionwise-paths-big-fixed-4-2runs/emd-positionwise-paths-big_0-fixed-4-2runs: 16302341 1223255 0.9302018031227012
# emd-positionwise-paths-big-fixed-4-3runs/emd-positionwise-paths-big_0: 16092161 1433435 0.9182090583395851
# emd-positionwise-paths-big-fixed-paths-2runs/emd-positionwise-paths-big_0: 16183977 1341619 0.9234480242497887
# emd-positionwise-paths-big-fixed-paths-3runs/emd-positionwise-paths-big_0: 16183179 1342417 0.9234024908482428

# SIM-ANNEAL:
# emd-positionwise-paths-big_0-fixed-4/emd-positionwise-paths-big_0:  16545051 980545 0.9440506902019196

# ONLY 4 POINTS:
#KAMADA:
# emd-positionwise-paths-big-ID-UN-AN-ST_0: 3 1 0.75
# emd-positionwise-paths-big-ID-UN-AN-ST_1: 3 1 0.75
# emd-positionwise-paths-big-ID-UN-AN-ST_2: 1 3 0.25
# emd-positionwise-paths-big-ID-UN-AN-ST_3: 4 0 1.0
# emd-positionwise-paths-big-ID-UN-AN-ST_4: 1 3 0.25
# emd-positionwise-paths-big-ID-UN-AN-ST_5: 4 0 1.0
# emd-positionwise-paths-big-ID-UN-AN-ST_6: 4 0 1.0
# emd-positionwise-paths-big-ID-UN-AN-ST_7: 3 1 0.75
# emd-positionwise-paths-big-ID-UN-AN-ST_8: 1 3 0.25
# emd-positionwise-paths-big-ID-UN-AN-ST_9: 4 0 1.0


#SIM-ANNEAL:
# emd-positionwise-paths-big-ID-UN-AN-ST_0: 4 0 1.0
# emd-positionwise-paths-big-ID-UN-AN-ST_1: 4 0 1.0
# emd-positionwise-paths-big-ID-UN-AN-ST_2: 4 0 1.0
# emd-positionwise-paths-big-ID-UN-AN-ST_3: 4 0 1.0
# emd-positionwise-paths-big-ID-UN-AN-ST_4: 4 0 1.0
# emd-positionwise-paths-big-ID-UN-AN-ST_5: 4 0 1.0
# emd-positionwise-paths-big-ID-UN-AN-ST_6: 4 0 1.0
# emd-positionwise-paths-big-ID-UN-AN-ST_7: 4 0 1.0
# emd-positionwise-paths-big-ID-UN-AN-ST_9: 4 0 1.0
