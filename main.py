import random
from collections import defaultdict
import numpy as np
import pandas as pd

from algorithms.kamadaY import KamadaY
from algorithms.kamada_kawai import KamadaKawai
from algorithms.kamada_xy import KamadaXY
from algorithms.simulated_ann import SimulatedAnnealing
from algorithms.spring_layout import SpringLayout
from algorithms.simulated_annealing import Bordawise
from evaluation.stability import StabilityEvaluation
from plots import plot_from_file, plot_using_map


def make_smaller_sample(file_path, save_name, only_selected_group=None, percent=0.2):
    df = pd.read_csv(file_path, sep=';')
    all_points_names = set(df['election_id_1'].tolist()) | set(df['election_id_2'].tolist())
    num_elections = len(all_points_names)

    grouped_points = defaultdict(list)

    for p in all_points_names:
        if '_' in p:
            group_name = p[:p.rfind("_")]
        else:
            group_name = p
        grouped_points[group_name].append(p)

    points_to_keep = {'Uniformity', 'Identity'}
    for k, v in grouped_points.items():
        if only_selected_group is not None:
            if k not in only_selected_group:
                continue

        points_to_keep.update(random.sample(v, int(len(v) * percent)))

    to_keep = []

    for i, row in df.iterrows():
        a_name = row['election_id_1']
        b_name = row['election_id_2']
        if a_name not in points_to_keep or b_name not in points_to_keep:
            to_keep.append(False)
        else:
            to_keep.append(True)

    df = df[to_keep]

    df.loc[df['distance'] < 0.0000001, 'distance'] = 0.00001

    df.to_csv(save_name, index=False, sep=';')


def generate_even_spaced_points(width, num_points):
    points = np.linspace(0, width, num_points)
    distances = []

    def _get_name(idx):
        if idx == len(points) - 1:
            return 'Uniformity'
        elif idx == 0:
            return 'Identity'
        return f'a_{idx}'

    le = len(points)

    for i in range(le):
        for j in range(i, le):
            p1 = points[i]
            p2 = points[j]
            distances.append(f'{_get_name(i)};{_get_name(j)};{abs(p1 - p2)};0.1234')

    with open('data/bordawise/even_spaced_points.csv', 'w') as f:
        f.write('election_id_1;election_id_2;distance;time')
        f.write('\n'.join(distances))


def main():
    b = Bordawise('data/bordawise/all-mallows-1d.csv')
    b.save('saved_all-mallows-1d')
    plot_from_file('saved_results/bordawise/saved_all-mallows-1d/test.csv')


def main_networkx():
    n = SpringLayout('data/bordawise/all-mallows-1d.csv')
    n.save('nx-all-mallows-1d')

    plot_from_file('saved_results/bordawise/nx-all-mallows-1d/test.csv')


def main_kamada_kawai():
    kamada_kawai = KamadaKawai('data/bordawise/mallows-unid-stun-stan-stid-3dsphere-3dcube.csv')
    saved_path = kamada_kawai.save()

    plot_from_file(saved_path)


def main_kamada_y():
    kamada_y = KamadaY('data/bordawise/mallows-unid-stun-stan-stid-3dsphere-3dcube.csv', max_neighbour_distance=3000)
    positions = kamada_y.get_positions()
    saved_path = positions.save()

    plot_from_file(saved_path)


def main_kamada_xy():
    kamada_xy = KamadaXY(
        'data/truncated_urn/distances/l1-approvalwise.csv',
        initial_positions_path='data/resampling/coordinates/kamada_xy/l1-approvalwise-full-ic-id-empty-bb-fixed.csv',
        optim_method='bb',
        fix_positions=True
        # max_neighbour_distance=100
    )

    positions = kamada_xy.get_positions()
    saved_path = positions.save(root_path='data/truncated_urn/coordinates/')

    plot_using_map(saved_path, 'data/truncated_urn/map.csv')


def main_sim_ann():
    ann = SimulatedAnnealing(
        'data/positionwise/emd-positionwise.csv',
        temperature=100000,
        num_stages=15,
        number_of_trials_for_temp=40,
        cooling_radius_factor=0.6,
        cooling_temp_factor=0.6
    )
    positions = ann.get_positions()
    saved_path = positions.save(root_path='saved_results/emd-positionwise')
    plot_from_file(saved_path)


def main_evaluate():
    file_path = 'data/positionwise/emd-positionwise-paths-big-ID-UN-AN-ST.csv'
    kamada_xy = KamadaXY(
        file_path,
        # initial_positions_path='saved_results/emd-positionwise/fixed_positions/emd-positionwise-paths-big-only-paths-bb.csv',
        optim_method='bb',
        epsilon=1e-20
        # max_neighbour_distance=100
    )
    positions = kamada_xy.get_positions()
    saved_path = positions.save(root_path='saved_results/emd-positionwise/temp')
    plot_from_file(saved_path)
    # evaluation = StabilityEvaluation(file_path)
    # evaluation.preprocess(kamada_xy)


def main_evaluate_sim():
    file_path = 'data/positionwise/emd-positionwise-paths-big.csv'
    sim = SimulatedAnnealing(
        file_path,
        initial_positions_path='saved_results/emd-positionwise/fixed_positions/emd-positionwise-paths-big-only-paths-bb.csv',
        temperature=100000,
        num_stages=15,
        number_of_trials_for_temp=40,
        cooling_radius_factor=0.6,
        cooling_temp_factor=0.6
    )
    evaluation = StabilityEvaluation(file_path)
    evaluation.preprocess(sim)

if __name__ == "__main__":
    # main_kamada_xy()
    # main_networkx()
    # plot_using_map('./saved_results/SOME_TEST/kamada_xy/emd-positionwise-paths-big-bb.csv', './map.csv')
    # generate_random_points(8000, 20)
    # make_smaller_sample(
    #     'data/resampling/distances/l1-approvalwise.csv',
    #     'data/resampling/distances/l1-approvalwise-full-ic-id-empty.csv',
    #     only_selected_group={
    #         'full',
    #         'IC 0.5',
    #         'empty',
    #         'ID 0.5',
    #     },
    #     percent=1.0
    # )
    main_kamada_xy()
    # plot_using_map('data/resampling/coordinates/kamada_xy/l1-approvalwise-bb.csv', 'data/resampling/map.csv')
    # plot_from_file('saved_results/bordawise/kamada_y/mallows-unid-stun-stan-stid-3dsphere-3dcube-kk-bb-top-close.csv')
    # b = Bordawise('small.csv')
    # b.save('saved_small')
    # b.plot()
    # plot_using_map('saved_results/emd-positionwise/kamada_xy/emd-positionwise-1000-bb-2steps.csv', 'map.csv')
