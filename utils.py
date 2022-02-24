from pathlib import Path

from algorithms.common import parse_data
import pandas as pd


def get_distances_to_point(distances, point_name, names_to_indexes, subgroups):
    indexes_to_take = []

    for name, i in names_to_indexes.items():
        if '_' in name:
            group_name = name[:name.rfind("_")]
        else:
            group_name = name

        if group_name in subgroups:
            indexes_to_take.append(i)

    point_index = names_to_indexes[point_name]
    return distances[point_index][indexes_to_take]


def test_distance(positions_file, distance_file):
    distances, num_elections, indexes_to_names = parse_data(distance_file, ['ID', 'UN', 'AN', 'ST'])

    an_dist = get_distances_to_point(
        distances,
        'AN',
        indexes_to_names,
        ['Norm-Mallows 0.5 (uniform)']
    )

    st_dist = get_distances_to_point(
        distances,
        'ST',
        indexes_to_names,
        ['Norm-Mallows 0.5 (uniform)']
    )

    print(f"AN sum: {an_dist.sum():.2f}, AN avg: {an_dist.mean():.2f} AN min: {an_dist.min():.2f}, AN max: {an_dist.max():.2f}\n"
          f"ST sum: {st_dist.sum():.2f}, ST avg: {st_dist.mean():.2f} ST min: {st_dist.min():.2f}, ST max: {st_dist.max():.2f}")


test_distance('saved_results/emd-positionwise/simulated_annealing/emd-positionwise-mallows-05.csv',
              'data/positionwise/emd-positionwise.csv')
