from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
import pandas as pd


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


def plot_using_map(file_path, map_path):
    df = pd.read_csv(file_path)
    df['point_type'] = df['names'].map(lambda x: x.split('_')[0])
    point_types = set(df['point_type'].tolist())

    map_df = pd.read_csv(map_path, sep=';')
    fig, ax = plt.subplots(figsize=(10, 6))

    for p_type in point_types:
        print(p_type)
        series = map_df[map_df['family_id'] == p_type].iloc[0]
        print(series)
        points = df[df['point_type'] == p_type]
        ax.scatter(points['x'], points['y'], label=series['label'], color=series['color'], marker=series['marker'], alpha=series['alpha'])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    save_root = Path(file_path).parent
    save_name = f"{Path(file_path).stem}.png"
    plt.tight_layout()
    plt.savefig(Path(save_root, save_name), dpi=300)
    plt.show()