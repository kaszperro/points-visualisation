from pathlib import Path

import networkx
import numpy as np
from algorithms.common import  VisualizationAlgorithm


class SpringLayout(VisualizationAlgorithm):

    def _get_positions(self, distances, num_elections):
        g = networkx.from_numpy_matrix(1 / (self.distances + 0.0001))

        positions = np.zeros((self.num_elections, 2))

        for i in range(self.num_elections):
            positions[i, 0] = self.distances[i, 0]
            if i > 1:
                positions[i, 1] = np.random.uniform(-5, 5)

        pos_to_dict = {
            i: positions[i] for i in range(self.num_elections)
        }

        for i, p in networkx.spring_layout(g, pos=pos_to_dict, iterations=1000).items():
            positions[i] = p

        return np.array(positions)

    def _get_save_file_name(self):
        return Path('networkx-spring-layout', f'{self.file_name}.csv')
