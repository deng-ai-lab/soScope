import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def L2_distance(x,y):
    return np.linalg.norm(x-y)


def get_distance_min_median(feature_file, filter_file, mode, dim, distance=L2_distance, start_index=0):
    """

    Args:
        feature_file: the path of image feature (for W) or sptail profile matrix (for A)
        filter_file: the position file in .csv to store the x-y position of each entry
        mode: the number of neighbors, users can only choose 4 or 6
        dim: the dimention of features used for calculate the distance
        distance: the distance metrics, default = L2_distance
        start_index: in case of a profile mixture, users can slice the input feature by adjusting start_index to select exact features.

    Returns:
        the minimum and median values of neighboring distance
    """
    gene = np.load(feature_file)[:, start_index:start_index+dim]
    print(gene.shape)
    merge_position = pd.read_csv(filter_file)
    position = [[merge_position['row'].tolist()[i], merge_position['col'].tolist()[i]] for i in range(len(merge_position['row'].tolist()))]
    print(len(position))

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

    min_ = np.min(index)
    median_ = np.median(index)
    return min_, median_


def get_weight_adj(feature_file, filter_file, mode,
                   dim, min_, median_, 
                   distance, scale=, start_index=0):
    """

    Args:
        feature_file: the path of image feature (for W) or sptail profile matrix (for A)
        filter_file: the position file in .csv to store the x-y position of each entry
        mode: the number of neighbors, users can only choose 4 or 6
        dim: the dimention of features used for calculate the distance
        min_: the minimum distance
        median_: the median distance
        distance: the distance metrics, default = L2_distance
        scale: a number >1. The median distance will have a similarity of -log(scale)
        start_index: in case of a profile mixture, users can slice the input feature by adjusting start_index to select exact features.

    Returns:
        A similarity matrix in the coo format.

    """
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
                dist_ = np.exp((-dist+lambda_)/delta_)
                index.append([i, idx_target, dist_])
    index = np.array(index)
    return index