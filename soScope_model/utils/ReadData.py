import numpy as np
import torch
from torch_geometric.utils import add_self_loops


def get_global_dataset(root_path):
    st_label_path = root_path + 'label.npy'
    st_label = np.load(st_label_path)
    node_num = st_label.shape[0]

    print(f'===There are {node_num} nodes to train.===')

    sub_spot = np.load(f'{root_path}/support_feature.npy')
    sub_adj = np.load(f'{root_path}/sub_adj.npy')

    sub_spot = torch.tensor(sub_spot).float()
    sub_edge = torch.tensor(sub_adj[:, :2].T).long()
    sub_edge_value = torch.tensor(sub_adj[:, 2].T).float()

    # Add self loop
    sub_edge, sub_edge_value = add_self_loops(sub_edge, sub_edge_value)

    tuple_ = [sub_spot, sub_edge, sub_edge_value, None]
    print("Sub Data Loaded")
    return [tuple_]


def get_global_bias_dataset(root_path):
    center_fea = np.load(f'{root_path}/sub_spot_bias.npy')
    adj = np.load(f'{root_path}/adj.npy')

    center_fea = torch.tensor(center_fea).float()
    edge = torch.tensor(adj[:, :2].T).long()
    edge_value = torch.tensor(adj[:, 2].T).float()
    # Add self loop
    edge, edge_value = add_self_loops(edge, edge_value)

    tuple_ = [center_fea, edge, edge_value, None]

    print("Center features Loaded")
    return [tuple_]


def get_global_spatial_dataset(root_path, mode):
    st_label_path = root_path + 'label.npy'
    print(f'We built edges between every neighboring {mode} nodes')
    adj_path = root_path + 'adj.npy'

    st_label = np.load(st_label_path)
    st_adj = np.load(adj_path)

    st_label = torch.tensor(st_label).float()
    st_adj_index = torch.tensor(st_adj[:, :2].T).long()
    st_adj_value = torch.tensor(st_adj[:, 2].T).float()
    # Add self loop
    st_adj_index_i, st_adj_value_i = add_self_loops(st_adj_index, st_adj_value)
    print("Spatial Data Loaded")
    return [st_label, st_adj_index_i, st_adj_value_i]


def get_global_expression_spatial_dataset(root_path, mode):
    st_label_path = root_path + 'bias_exp.npy'
    print(f'We built edges between every neighboring {mode} nodes')
    bias_exp = np.load(st_label_path)
    bias_exp = torch.tensor(bias_exp).float()
    print("Bias Spatial Data Loaded")
    return bias_exp
