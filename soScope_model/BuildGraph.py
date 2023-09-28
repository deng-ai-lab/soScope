import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv


def L2_distance(x,y):
    return np.linalg.norm(x-y)


def get_distance_mu_std_var(name, filter_file, feature_file, mode, dim, distance, start_index=0):
    gene = np.load(feature_file)[:, start_index:start_index+dim]
    print(gene.shape)
    merge_position = pd.read_csv(filter_file)
    position = [[merge_position['row'].tolist()[i], merge_position['col'].tolist()[i]] for i in range(len(merge_position['row'].tolist()))]
    print(len(position))

    # mode = 4  # 选择4邻域，即认为4邻域内的点为邻居
    # mode = 6  # 选择6邻域，即认为6邻域内的点为邻居 （Visium Only）

    offsets = []
    if mode == 4:
        offsets = [[-1, 0],
                   [0, -1], [0, +1],
                   [+1, 0]]
    if mode == 6:
        offsets = [[-1, -1], [-1, +1],
                   [0, -2], [0, +2],
                   [+1, -1], [+1, +1]]
    assert offsets != [], 'NO MODE SELECTED!'

    index = []

    for i in range(len(position)):
        for offset in offsets:
            neighbor_x = position[i][0] + offset[0]
            neighbor_y = position[i][1] + offset[1]
            if [neighbor_x, neighbor_y] in position:
                idx_target = position.index([neighbor_x, neighbor_y])
                dist = distance(gene[i], gene[idx_target])
                index.append(dist)

    plt.hist(index, bins=50, color="blue", label=distance.__name__, density=True, histtype="bar", edgecolor='white')
    plt.title(name + " Distance Distribution", fontdict={'fontsize': 18, 'color': 'r'})
    plt.xlabel(distance.__name__)
    plt.ylabel("Frequency")  # 显示图例plt.legend(loc = 'best')
    # 显示图形
    plt.show()

    min_ = np.min(index)
    median_ = np.median(index)
    return min_, median_


def get_weight_adj(filter_file, feature_file, mode,
                   dim, min_, median_, 
                   dist_mode, distance, scale=2, start_index=0):
    gene = np.load(feature_file)[:, start_index:start_index+dim]
    merge_position = pd.read_csv(filter_file)
    position = [[merge_position['row'].tolist()[i], merge_position['col'].tolist()[i]] for i in range(len(merge_position['row'].tolist()))]

    offsets = []
    if mode == 4:
        offsets = [[-1, 0],
                   [0, -1],[0, +1],
                   [+1, 0]]
    if mode == 6:
        offsets = [[0, +2], [+1, +1], [+1, -1],
                   [0, -2], [-1, -1], [-1, +1]]
    assert offsets != [], 'NO MODE SELECTED!'

    index = []
    delta_ = (median_ - min_)/np.log(scale)
    lambda_ = min_
    for i in range(len(position)):
        for offset in offsets:
            neighbor_x = position[i][0] + offset[0]
            neighbor_y = position[i][1] + offset[1]
            if [neighbor_x, neighbor_y] in position:
                idx_target = position.index([neighbor_x, neighbor_y])
                dist = distance(gene[i], gene[idx_target])
                if dist_mode == 'L2':
                    dist_ = dist
                elif dist_mode == 'exp':
                    dist_ = np.exp((-dist+lambda_)/delta_)
                index.append([i, idx_target, dist_])
    index = np.array(index)
    return index