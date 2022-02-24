from pathlib import Path

from algorithms.common import parse_data, save_positions
from fa2 import ForceAtlas2
import numpy as np


class Force:
    def __init__(self, file_path):
        self.distances, self.num_elections, self.indexes_to_names = parse_data(file_path)
        self.file_name = Path(file_path).stem
        self.positions = self.get_positions()

    def get_positions(self):
        fa = ForceAtlas2(gravity=1.0, edgeWeightInfluence=5)

        positions = np.zeros((self.num_elections, 2))

        for i in range(self.num_elections):
            positions[i, 0] = self.distances[i, 0]
            if i > 1:
                positions[i, 1] = np.random.uniform(-50, 50)

        pos = fa.forceatlas2(
            -self.distances,
            positions,
            iterations=10000,
            fixed_positions_indexes={0, 1},
            fix_x=True
        )

        return pos

    def save(self, root_path='saved_results'):
        p = Path(root_path, 'forceatlas', f'{self.file_name}.csv')
        save_positions(p, self.positions, self.indexes_to_names)
