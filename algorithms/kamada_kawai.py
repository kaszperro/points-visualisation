from pathlib import Path

import networkx as nx
import numpy as np

from algorithms.common import parse_data, save_positions


class KamadaKawai:
    def __init__(self, file_path):
        self.distances, self.num_elections, self.indexes_to_names = parse_data(file_path)
        self.file_name = Path(file_path).stem
        self.positions = self.get_positions()

    def get_positions(self):
        g = nx.from_numpy_matrix(self.distances)

        positions = np.zeros((self.num_elections, 2))

        for i in range(self.num_elections):
            positions[i, 0] = self.distances[i, 0]
            if i > 1:
                positions[i, 1] = np.random.uniform(-1, 1)

        pos_to_dict = {
            i: positions[i] for i in range(self.num_elections)
        }

        dist_to_dict = {
            i: {
                j: self.distances[i, j] for j in range(self.num_elections) if i != j
            }
            for i in range(self.num_elections)
        }

        positions_dict = nx.kamada_kawai_layout(
            g,
            dist=dist_to_dict,
            # pos=pos_to_dict
        )

        for k, v in positions_dict.items():
            positions[k] = v
        return positions

    def save(self, root_path='saved_results'):
        p = Path(root_path, 'kamada_kawai', f'{self.file_name}.csv')
        save_positions(p, self.positions, self.indexes_to_names)
        return p
