from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd


def _read_data(path):
    df = pd.read_csv(path, sep=';')
    return df


# x[0, ...] = Identity
# x[1, ...] = Uniform
def parse_data(file_path, special_names_first=None):
    if special_names_first is None:
        special_names_first = []

    data = _read_data(file_path)
    all_points_names = set(data['election_id_1'].tolist()) | set(data['election_id_2'].tolist())
    num_elections = len(all_points_names)
    x = np.zeros((num_elections, num_elections))

    indexes = {
        type_name: i for i, type_name in enumerate(special_names_first)
    }
    idx = len(special_names_first)

    for i, row in data.iterrows():
        a_name = row['election_id_1']
        b_name = row['election_id_2']
        dist = row['distance']

        if a_name not in indexes:
            indexes[a_name] = idx
            idx += 1

        if b_name not in indexes:
            indexes[b_name] = idx
            idx += 1
        i1 = indexes[a_name]
        i2 = indexes[b_name]

        x[i1, i2] = dist
        x[i2, i1] = dist

    assert idx == num_elections

    return x, num_elections, indexes


def save_positions(save_path, positions, indexes_to_names):
    num_elections = len(positions)
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    df_positions = pd.DataFrame(data=positions, columns=['x', 'y'])

    names = ['' for _ in range(num_elections)]
    for k, v in indexes_to_names.items():
        names[v] = k

    df_positions['names'] = names
    df_positions.to_csv(save_path)


class Positions:
    def __init__(self, positions, names_to_indexes, file_name):
        self.file_name = file_name

        num_elections = len(positions)
        self.df_positions = pd.DataFrame(data=positions, columns=['x', 'y'])

        names = ['' for _ in range(num_elections)]
        for name, index in names_to_indexes.items():
            names[index] = name

        self.df_positions['names'] = names

    def save(self, root_path=Path('saved_results/bordawise'), overridden_file_name=None):
        if overridden_file_name is not None:
            file_name = overridden_file_name
        else:
            file_name = self.file_name

        save_path = Path(root_path, file_name)

        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.df_positions.to_csv(save_path)

        return save_path


class VisualizationAlgorithm(ABC):
    def __init__(self, file_path, fixed_positions_path=None):
        self.file_path = file_path
        self.fixed_positions_path = fixed_positions_path

        self.reload()

    def reload(self):
        if self.fixed_positions_path is not None:
            self.fixed_positions_indexes, self.fixed_positions = self._read_fixed_positions(self.fixed_positions_path)
        else:
            self.fixed_positions_indexes, self.fixed_positions = None, None

        self.distances, self.num_elections, self.names_to_indexes = parse_data(
            self.file_path,
            self.fixed_positions_indexes
        )
        self.file_name = Path(self.file_path).stem

        self._reload()

    def _reload(self):
        raise NotImplementedError

    def _read_fixed_positions(self, path):
        df = pd.read_csv(path)
        return df['names'].tolist(), df[['x', 'y']].to_numpy()

    def _apply_fixed_positions(self, positions):
        if self.fixed_positions is not None:
            for i in range(len(self.fixed_positions)):
                positions[i] = self.fixed_positions[i]

    def _get_point_index(self, point_possible_names):
        for key, val in self.names_to_indexes.items():
            if key in point_possible_names:
                return val

    def _initial_place_on_circle(self):
        identity_index = self._get_point_index(['Sym', 'Identity'])
        uniformity_index = self._get_point_index(['Asym', 'Uniformity'])

        identity_uniformity_distance = self.distances[identity_index, uniformity_index]

        positions = circle_points(
            [identity_uniformity_distance / 2, identity_uniformity_distance * 2],
            [self.num_elections // 2, self.num_elections - self.num_elections // 2]
        )

        self._apply_fixed_positions(positions)

        return positions

    def _get_positions(self, distances, num_elections):
        raise NotImplementedError()

    def _get_save_file_name(self):
        raise NotImplementedError()

    def get_positions(self):
        positions = self._get_positions(self.distances, self.num_elections)
        return Positions(positions, self.names_to_indexes, self._get_save_file_name())


def _place_on_circle_2d(distances, num_elections):
    r = distances[0, 1] / 2
    cent_x = r
    rs = r ** 2

    positions = np.zeros((num_elections, 2))
    for i in range(num_elections):
        x = distances[i, 0]
        y = 0
        if i > 1:
            y = np.sqrt(rs - (x - cent_x) ** 2) / 2
            # if i % 2 == 0:
            #     y *= -1
        positions[i] = [x, y]
    return positions


def circle_points(r, n):
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.extend(np.c_[x, y])
    return np.array(circles)


def _calc_k_with_special_value(distances, special_value, indexes=None):
    square_dist = distances ** 2
    np.fill_diagonal(square_dist, 1)
    _close_zero(square_dist)
    k = np.ones_like(square_dist)

    if indexes is not None:
        for i in indexes:
            k[:, i] = special_value
            k[i, :] = special_value

    k = k / square_dist
    np.fill_diagonal(k, 0)

    return k


def _close_zero(matrix, eps=1e-5):
    cond = matrix < eps
    matrix[cond] = eps

    return matrix


def _upper_tri_sum(matrix):
    return np.triu(matrix, 1).sum()
