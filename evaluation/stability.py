import time
from collections import defaultdict
from pathlib import Path
import random

import pandas as pd
import numpy as np

from algorithms.common import VisualizationAlgorithm
from evaluation.utils import EvaluationAlgorithm
import shutil

import matplotlib.pyplot as plt

from plots import plot_using_map


def _shuffle_csv(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    header, rest = lines[0], lines[1:]
    random.shuffle(rest)

    shuffled_file = [header] + rest

    with open(csv_path, 'w') as f:
        f.write(''.join(shuffled_file))


def get_id_un_distance(df):
    id_name = 'Identity' if 'Identity' in df['names'].values else 'ID'
    un_name = 'Uniformity' if 'Uniformity' in df['names'].values else 'UN'

    id_row = df[df['names'] == id_name].iloc[0]
    id_pos = np.array([id_row['x'], id_row['y']])

    un_row = df[df['names'] == un_name].iloc[0]
    un_pos = np.array([un_row['x'], un_row['y']])

    return np.linalg.norm(id_pos - un_pos)


class StabilityEvaluation(EvaluationAlgorithm):

    def __init__(self, file_path):
        original_file = Path(file_path)
        self.file_name = original_file.stem
        self.temp_file_path = Path('evaluation', 'saved', 'stability', 'tmp', f'tmp_{self.file_name}.csv')
        self.temp_file_path.parent.mkdir(exist_ok=True, parents=True)

        shutil.copyfile(original_file, self.temp_file_path)

        self.save_results_root = Path('evaluation', 'saved', 'stability', 'results', f'{self.file_name}')
        self.save_results_root.mkdir(exist_ok=True, parents=True)

    def preprocess(self, visualisation_algorithm: VisualizationAlgorithm):
        visualisation_algorithm.file_path = self.temp_file_path

        times_took = []

        for i in range(10):
            print(f"RUN {i + 1}/10")
            _shuffle_csv(self.temp_file_path)
            visualisation_algorithm.reload()
            time_start = time.time()
            positions = visualisation_algorithm.get_positions()
            time_end = time.time()

            times_took.append(time_end - time_start)

            positions.save(self.save_results_root, overridden_file_name=f'{self.file_name}_{i}.csv')

        pd.Series(times_took, name='time').to_csv(Path(self.save_results_root, 'stats.csv'), index=False)

        return self.save_results_root


    def calculate_from_saved_path(self, root_path):
        root_path = Path(root_path)
        csv_paths = root_path.glob('*.csv')

        calculated_positions = defaultdict(list)
        id_un_distance = 0
        for path in csv_paths:
            if path.name != 'stats.csv':
                df = pd.read_csv(path)

                for i, row in df.iterrows():
                    calculated_positions[row['names']].append(np.array([row['x'], row['y']]))
                id_un_distance = get_id_un_distance(df)

        distances = defaultdict(list)

        for key, values in calculated_positions.items():
            for v in values[1:]:
                distances[key].append(np.linalg.norm(v - values[0]))

        mean_distances = []
        max_distances = []

        for key, values in distances.items():
            mean_distance = np.mean(values)
            max_distance = np.max(values)
            mean_distances.append(mean_distance)
            max_distances.append(max_distance)

        plt.hist(max_distances, bins=30)
        plt.show()


# StabilityEvaluation.calculate_from_saved_path('saved/stability/results/emd-positionwise-paths-big')


# plot_using_map('saved/stability/results/emd-positionwise-paths-big-fixed-paths-3runs/emd-positionwise-paths-big_9.csv', '../map.csv')