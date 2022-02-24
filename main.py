import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
import numpy as np
import pandas as pd

from algorithms.forceatlas import Force
from algorithms.kamadaY import KamadaY
from algorithms.kamada_kawai import KamadaKawai
from algorithms.kamada_xy import KamadaXY
from algorithms.simulated_ann import SimulatedAnnealing
from algorithms.spring_layout import SpringLayout
from algorithms.simulated_annealing import Bordawise


def _generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(
        math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(
        number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades,
                                                          number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    print(number_of_partitions)

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        if lower_half > 0:
            initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8 / lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (
                    j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier
    # return initial_cm.reshape(16, 7, 3)
    return ListedColormap(initial_cm)


def plot_from_file(file_path):
    df = pd.read_csv(file_path)
    df['point_type'] = df['names'].map(lambda x: x.split('_')[0])

    colors = ['black', 'red', 'gold', 'blue', 'lime', 'magenta']
    print(len(colors))
    marker_styles = ['.', 'v', 's', '1', '*']

    point_types = set(df['point_type'].tolist())

    color_i = 0
    marker_i = 0

    point_types_styles = {}

    for i, point_type in enumerate(point_types):
        point_types_styles[point_type] = {
            'color': colors[color_i],
            'marker': marker_styles[marker_i]
        }

        color_i += 1
        if color_i >= len(colors):
            color_i = 0
            marker_i += 1

    fig, ax = plt.subplots(figsize=(10, 6))

    for p_type in point_types:
        style = point_types_styles[p_type]
        points = df[df['point_type'] == p_type]
        ax.scatter(points['x'], points['y'], label=p_type, color=style['color'], marker=style['marker'])

    # Now this is actually the code that you need, an easy fix your colors just cut and paste not you need ax.
    # colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
    # colorst = [colormap(i) for i in np.linspace(0, 0.9, len(ax.collections))]
    # for t, j1 in enumerate(ax.collections):
    #     j1.set_color(colorst[t])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    save_root = Path(file_path).parent
    save_name = f"{Path(file_path).stem}.png"
    plt.tight_layout()
    plt.savefig(Path(save_root, save_name), dpi=300)
    plt.show()


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


def main_force():
    f = Force('data/bordawise/all-mallows-1d.csv')
    f.save('force-all-mallows-1d')

    plot_from_file('saved_results/bordawise/force-all-mallows-1d/test.csv')


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
        'data/positionwise/emd-positionwise-paths-big.csv',
        fixed_positions_path='saved_results/emd-positionwise/fixed_positions/emd-positionwise-paths-big-ID-UN-AN-ST-bb.csv',
        optim_method='bb',
        # max_neighbour_distance=100
    )

    # kamada_xy = KamadaXY(
    #     'data/positionwise/emd-positionwise-paths-big-ID-UN-AN-ST.csv',
    #     # fixed_positions_path='saved_results/emd-positionwise/fixed_positions/emd-positionwise-paths-big-ID-UN-AN-ST-bb.csv',
    #     optim_method='bb'
    # )
    positions = kamada_xy.get_positions()
    saved_path = positions.save(root_path='saved_results/emd-positionwise/my_test')

    plot_from_file(saved_path)


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


# x coś tam zrobić
if __name__ == "__main__":
    # main_networkx()
    # generate_random_points(8000, 20)
    # make_smaller_sample(
    #     'data/positionwise/emd-positionwise-1000.csv',
    #     'data/positionwise/emd-positionwise-1000-ID-UN-AN-ST.csv',
    #     only_selected_group={
    #         'Identity',
    #         'Uniformity',
    #         'Antagonism',
    #         'Stratification',
    #         # 'ANUN',
    #         # 'ANID',
    #         # 'STUN',
    #         # 'STID',
    #         # 'UNID',
    #         # 'STAN'
    #     },
    #     percent=1.0
    # )
    main_kamada_xy()
    # plot_from_file('saved_results/bordawise/kamada_y/mallows-unid-stun-stan-stid-3dsphere-3dcube-kk-bb-top-close.csv')
    # b = Bordawise('small.csv')
    # b.save('saved_small')
    # b.plot()
