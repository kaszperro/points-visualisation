from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.common import VisualizationAlgorithm, parse_data
from evaluation.utils import EvaluationAlgorithm
from plots import plot_using_importance_matrix, plot_using_map


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


# point_name_to_d = Distortion('../data/positionwise/emd-positionwise-paths-big.csv', 0.1).calculate_from_saved_path(
#     'saved/kamda_bb/stability/results/emd-positionwise-paths-big-fixed-paths-3runs')
#
# plot_using_importance_matrix(
#     'saved/kamda_bb/stability/results/emd-positionwise-paths-big-fixed-paths-3runs/emd-positionwise-paths-big_0.csv', point_name_to_d)


for i in range(1):
    csv_path = Path(f'../saved_results/SOME_TEST/kamada_xy/emd-positionwise-paths-big-bb.csv')
    print(csv_path.stem)
    point_name_to_m = Distortion('../data/positionwise/emd-positionwise-paths-big.csv').calculate_from_saved_path(
        csv_path)

    plot_using_importance_matrix(csv_path, point_name_to_m, show=False, save_path=Path(csv_path.parent, f'{csv_path.stem}_dist.png'))
    plot_using_map(csv_path, '../map.csv', False)


#MEAN DIST (10%):
# KAMADA KAWAI:
#emd-positionwise-paths-big_0-fixed-4-2runs.csv : 1.0498735726898416
#emd-positionwise-paths-big-fixed-4-3runs/emd-positionwise-paths-big_0.csv :  1.054683614829009
#emd-positionwise-paths-big-fixed-paths-2runs/emd-positionwise-paths-big_0: 1.0523417299172935
#emd-positionwise-paths-big-fixed-paths-3runs/emd-positionwise-paths-big_0: 1.0525321459154457
# SIM-ANNEAL:
# emd-positionwise-paths-big_0-fixed-4/emd-positionwise-paths-big_0: 1.240033563466185



# ONLY 4 POINTS:
# KAMADA-KAWAI:
# emd-positionwise-paths-big-ID-UN-AN-ST_0: 1.1124199634308398
# emd-positionwise-paths-big-ID-UN-AN-ST_1: 1.1124199633445295
# emd-positionwise-paths-big-ID-UN-AN-ST_2: 1.4497158678654567
# emd-positionwise-paths-big-ID-UN-AN-ST_3: 1.0033169247768923
# emd-positionwise-paths-big-ID-UN-AN-ST_4: 1.4497158678654567
# emd-positionwise-paths-big-ID-UN-AN-ST_5: 1.0033169247767093
# emd-positionwise-paths-big-ID-UN-AN-ST_6: 1.0033169247768923
# emd-positionwise-paths-big-ID-UN-AN-ST_7: 1.1124199633446914
# emd-positionwise-paths-big-ID-UN-AN-ST_8: 1.4497158678649327
# emd-positionwise-paths-big-ID-UN-AN-ST_9: 1.0033169247768923


#SIM-ANNEAL:

# emd-positionwise-paths-big-ID-UN-AN-ST_0: 1.0042275847243423
# emd-positionwise-paths-big-ID-UN-AN-ST_1: 1.0040229696124507
# emd-positionwise-paths-big-ID-UN-AN-ST_2: 1.0035342431385024
# emd-positionwise-paths-big-ID-UN-AN-ST_3: 1.0043248909762537
# emd-positionwise-paths-big-ID-UN-AN-ST_4: 1.004129188840956
# emd-positionwise-paths-big-ID-UN-AN-ST_5: 1.0043893743722525
# emd-positionwise-paths-big-ID-UN-AN-ST_6: 1.003733727225542
# emd-positionwise-paths-big-ID-UN-AN-ST_7: 1.0038813442043888
# emd-positionwise-paths-big-ID-UN-AN-ST_8: 1.0039187073184674
# emd-positionwise-paths-big-ID-UN-AN-ST_9: 1.0042489165886128
