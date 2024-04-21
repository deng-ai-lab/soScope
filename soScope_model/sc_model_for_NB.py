import torch
import torch.nn as nn
import torch.nn.functional as F
from ..training.soScope_model_for_Gaussian import soScope_Gaussian
from ..utils.modules import Nb_NodeDecoder
from torch.distributions import Independent, NegativeBinomial
import random
import numpy as np

def dot_product_decode(z):
    adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
    return adj_pred

# def median_normalize(count_matrix):
#     median_factor, _ = torch.median(count_matrix, dim=0)
#     adjust_factor = median_factor + 1
#     normalized_count = count_matrix/adjust_factor
#     return normalized_count, adjust_factor


def block_sum_sc(input_tensor, mask):
    """
    Sum up every n rows of input
    Args:
        input_tensor: a tensor with dimension of [m*n,d]
        mask: mask indicating how to merge

    Returns:
        a tensor with dimension of [m,d]
    """
    output = torch.mm(mask, input_tensor)
    return output



class sc_NB(soScope_Gaussian):
    def __init__(self, sub_node, **params):
        super(sc_NB, self).__init__(**params)
        self.sub_node = sub_node
        self.decoder_x = Nb_NodeDecoder(self.z_dim, self.gene_dim)
        self.mask_array = np.load('DataSet/sc/mask.npy')
        self.mask = torch.from_numpy(self.mask_array).float().to('cuda')

    def decode(self, sub_z, subplot):
        sub_z_ = torch.mm(self.mask.t(), sub_z)
        sub_log_p, sub_r = self.decoder(sub_z_, subplot)
        p_epsilon_given_z_f = Independent(NegativeBinomial(total_count=sub_r, logits=sub_log_p), 1)
        return p_epsilon_given_z_f


    def infer(self, st_data, sub_data, nonnegative=True, mode='sum'):
        assert mode == 'sum' or 'mean', 'No such mode, only sum and mean available'
        self.eval()

        x_, edge_index = st_data[:2]
        x = x_[:, :self.gene_dim]
        # x_input, _ = median_normalize(x * self.sub_node)
        # x_input, _, _ = min_max_normalize(x * self.sub_node)
        x_input = x.float()
        subplot_, sub_edge, sub_edge_value, index = sub_data
        subplot = subplot_[:, :self.sub_dim]
        device = self.get_device()
        self.mask = torch.from_numpy(self.mask_array).float().to(device)

        p_z_given_x_a = self.encode(x_input, edge_index)

        z = p_z_given_x_a.mean
        p_epsilon_given_z_f = self.decode(z, subplot)

        epsilon = p_epsilon_given_z_f.mean
        if nonnegative:
            sub_x_pred = F.relu(epsilon)
        else:
            sub_x_pred = epsilon
        return sub_x_pred.detach()

    def estimate(self, st_data, sub_datas, mode='mean'):
        assert mode == 'sum' or 'mean', 'No such mode, only sum and mean available'
        device = self.get_device()

        for i, item in enumerate(st_data):
            st_data[i] = item.to(device)

        for sub_data in sub_datas:
            for j, item in enumerate(sub_data):
                if sub_data[j] is not None:
                    sub_data[j] = item.to(device)

        average_sq_error = 0
        count = 0
        x_ = st_data[0]
        x = x_[:, :self.gene_dim]

        for sub_data in sub_datas:
            count += 1
            sub_x_pred = self.infer(st_data, sub_data)
            mean_x_pred = block_sum_sc(sub_x_pred, self.mask)
            average_sq_error = average_sq_error + torch.dist(mean_x_pred, x, p=2).item()
        average_sq_error = average_sq_error/count
        return average_sq_error

    def _compute_loss(self, st_data, sub_data, mode='mean'):
        assert mode == 'sum' or 'mean', 'No such mode, only sum and mean available'
        x_, edge_index, edge_value = st_data

        x = x_[:, :self.gene_dim]
        x_all = x.long()
        x_input = x.float()

        subplot_, sub_edge, sub_edge_value, index = sub_data
        subplot = subplot_[:, :self.sub_dim]

        subspot_num = subplot.size()[-2]

        p_z_given_x_a = self.encode(x_input, edge_index)
        # Sample from the posteriors with reparametrization
        z = p_z_given_x_a.rsample()
        # z = p_z_given_x_a.mean

        p_epsilon_given_z_f = self.decode(z, subplot)
        epsilon = p_epsilon_given_z_f.mean  # [node_num, gene_dim]
        # norm_ep = (epsilon - min_)/factor
        sub_x_truth = x_all
        # sub_x_float = x_all.float()

        mse_loss = nn.MSELoss(reduction='mean')
        node_loss_2 = (1 / self.scale) * mse_loss(
            (block_sum_sc(epsilon, self.mask)).view(sub_x_truth.shape[0], -1),
            x_input.view(sub_x_truth.shape[0], -1))
        node_loss = node_loss_2.mean()

        # KL Divergence
        # the meaning is to minimize -Ep(z|x,A)[log p(z|x,A) - log p(z)], that is KL[p(z|x,A)||p(z)]
        kl_loss_1 = p_z_given_x_a.log_prob(z) - self.z_prior.log_prob(z)
        kl_loss = kl_loss_1.mean()

        # Predict Adjacency Matrix
        sub_adj_pred = dot_product_decode(epsilon - torch.mean(epsilon, dim=0))
        # sub_adj_pred = sparse_product_decode(epsilon - x.repeat_interleave(self.sub_node, dim=0), edge_index)
        # graph_loss_ = F.binary_cross_entropy(sub_adj_pred.view(-1), edge_value.view(-1))
        # adj_label = torch.sparse.FloatTensor(edge_index, edge_value, torch.Size([spot_num, spot_num]))
        sub_edge = torch.sparse.FloatTensor(sub_edge, sub_edge_value, torch.Size([subspot_num, subspot_num]))

        graph_loss_ = F.binary_cross_entropy(sub_adj_pred.view(-1), sub_edge.to_dense().view(-1))
        graph_loss = graph_loss_.sum()

        beta = self.beta_scheduler(self.iterations)

        # Logging the components
        self._add_loss_item('loss/kl_loss', kl_loss.item())
        self._add_loss_item('loss/node_loss', node_loss.item())
        self._add_loss_item('loss/graph_loss', graph_loss.item())
        self._add_loss_item('loss/beta', beta)

        # Computing the loss function
        loss = node_loss + self.graph_loss_weight * graph_loss + beta * kl_loss
        return loss
