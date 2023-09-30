import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, NegativeBinomial, Poisson
from ..utils.modules import JointSubGraphEncoder_Poisson_NB as EnhanceDecoder
from ..utils.modules import JointNodeDecoder_Poisson_NB
from ..utils.modules import GraphEncoder
from ..training.soScope_model_for_Gaussian import soScope_Gaussian

from ..training.base import init_optimizer, Trainer

def dot_product_decode(z):
    adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
    return adj_pred


def sparse_product_decode(z, st_index):
    index_s = st_index[0, :]
    index_t = st_index[1, :]
    z_s = torch.index_select(z, 0, index_s)
    z_t = torch.index_select(z, 0, index_t)
    inner_prod = torch.sum(z_s * z_t, dim=1)
    return torch.sigmoid(inner_prod)

def block_sum(input_tensor, n):
    """
    Sum up every n rows of input
    Args:
        input_tensor: a tensor with dimension of [m*n,d]
        n: number of rows to sum

    Returns:
        a tensor with dimension of [m,d]
    """
    device = input_tensor.device
    sum_uint = torch.ones((1, n), device=device)
    sum_mat = torch.block_diag(sum_uint)
    m = input_tensor.size()[-2] // n
    for i in range(m - 1):
        sum_mat = torch.block_diag(sum_mat, sum_uint)
    sum_mat = sum_mat.to(device)
    output = torch.mm(sum_mat, input_tensor)
    return output

class soScope_Joint(soScope_Gaussian):
    def __init__(self, protein_dim, sub_node, **params):
        super(soScope_Joint, self).__init__(**params)

        self.protein_dim = protein_dim
        self.sub_node = sub_node
        # Intialization of the encoder
        self.encoder_st = GraphEncoder(self.protein_dim + self.gene_dim, self.z_dim, head=self.head_num)
        self.decoder = EnhanceDecoder(self.z_dim, self.sub_dim, self.protein_dim, self.gene_dim)
        self.decoder_x = JointNodeDecoder_Poisson_NB(self.z_dim, self.protein_dim, self.gene_dim)

        self.opt = init_optimizer(self.optimizer_name, [
            {'params': self.encoder_st.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay},
            {'params': self.decoder.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay},
        ])

    def _get_items_to_store(self):
        items_to_store = super(soScope_Joint, self)._get_items_to_store()
        items_to_store['encoder_st'] = self.encoder_st.state_dict()
        items_to_store['decoder'] = self.decoder.state_dict()
        return items_to_store

    def decode(self, sub_z, subplot):
        ##########
        # Branch #
        ##########
        sub_z = sub_z.repeat_interleave(self.sub_node, dim=0)
        lambda_, sub_r, sub_log_p = self.decoder(sub_z, subplot)
        protein_hat_given_z_f = Independent(Poisson(rate=lambda_), 1)
        gene_hat_given_z_f = Independent(NegativeBinomial(total_count=sub_r, logits=sub_log_p), 1)
        return protein_hat_given_z_f, gene_hat_given_z_f

    def infer(self, st_data, sub_data, nonnegative=True, mode='mean'):
        assert mode == 'sum' or 'mean', 'No such mode, only sum and mean available'
        self.eval()
        x_, edge_index = st_data[:2]
        subplot_, sub_edge, sub_edge_value, index = sub_data
        subplot = subplot_[:, :self.sub_dim]

        p_z_given_x_a = self.encode(x_, edge_index)
        z = p_z_given_x_a.mean
        protein_hat_given_z_f, gene_hat_given_z_f = self.decode(z, subplot)
        epsilon_p = protein_hat_given_z_f.mean
        epsilon_g = gene_hat_given_z_f.mean

        epsilon = torch.cat([epsilon_p, epsilon_g], dim=1)
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

        for sub_data in sub_datas:
            count += 1
            sub_x_pred = self.infer(st_data, sub_data)
            sum_p_pred = block_sum(sub_x_pred[:, :self.protein_dim], self.sub_node)
            sum_g_pred = block_sum(sub_x_pred[:, self.protein_dim:], self.sub_node)
            sum_x_pred = torch.cat([sum_p_pred, sum_g_pred], dim=1)
            average_sq_error = average_sq_error + torch.dist(sum_x_pred, x_*self.sub_node, p=2).item()
        average_sq_error = average_sq_error/count
        return average_sq_error

    def _train_step(self, st_data, sub_data, bias_data=None):
        loss = self._compute_loss(st_data, sub_data)
        gamma = self.gamma_scheduler(self.iterations)
        loss = gamma * loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def _compute_loss(self, st_data, sub_data, mode='mean'):
        assert mode == 'sum' or 'mean', 'No such mode, only sum and mean available'
        x_, edge_index, edge_value = st_data
        p_mean = x_[:, :self.protein_dim]
        g_mean = x_[:, self.protein_dim:]
        p_truth = x_[:, :self.protein_dim] * self.sub_node
        g_truth = x_[:, self.protein_dim:] * self.sub_node

        subplot_, sub_edge, sub_edge_value, index = sub_data
        subplot = subplot_[:, :self.sub_dim]
        subspot_num = subplot.size()[-2]


        p_z_given_x_a = self.encode(x_, edge_index)
        # Sample from the posteriors with reparametrization
        z = p_z_given_x_a.rsample()
        protein_hat_given_z_f, gene_hat_given_z_f = self.decode(z, subplot)
        epsilon_p = protein_hat_given_z_f.mean  # [node_num, protein_dim]
        epsilon_g = gene_hat_given_z_f.mean  # [node_num, protein_dim]


        mse_loss = nn.MSELoss(reduction='mean')
        sum_epsilon_p = block_sum(epsilon_p, self.sub_node)
        sum_epsilon_g = block_sum(epsilon_g, self.sub_node)
        node_loss_p = (1/self.scale) * \
                      mse_loss(sum_epsilon_p.view(p_truth.shape[0], -1),
                               p_truth.view(p_truth.shape[0], -1))
        node_loss_g = (1/self.scale) * \
                      mse_loss(sum_epsilon_g.view(g_truth.shape[0], -1),
                               g_truth.view(g_truth.shape[0], -1))
        node_loss = node_loss_p.mean() + node_loss_g.mean()

        # node_loss = node_loss_2.mean()
        # KL Divergence
        # the meaning is to minimize -Ep(z|x,A)[log p(z|x,A) - log p(z)], that is KL[p(z|x,A)||p(z)]
        kl_loss_1 = p_z_given_x_a.log_prob(z) - self.z_prior.log_prob(z)
        kl_loss = kl_loss_1.mean()

        # Predict Adjacency Matrix
        sub_adj_pred_p = dot_product_decode(epsilon_p - p_mean.repeat_interleave(self.sub_node, dim=0))
        sub_adj_pred_g = dot_product_decode(epsilon_g - g_mean.repeat_interleave(self.sub_node, dim=0))
        sub_adj_pred = sub_adj_pred_p + sub_adj_pred_g - sub_adj_pred_p * sub_adj_pred_g

        sub_edge_value_ = torch.sparse.FloatTensor(sub_edge, sub_edge_value, torch.Size([subspot_num, subspot_num]))
        graph_loss_2 = F.binary_cross_entropy(sub_adj_pred.view(-1), sub_edge_value_.to_dense().view(-1))

        graph_loss = self.graph_loss_weight * graph_loss_2.sum()

        beta = self.beta_scheduler(self.iterations)

        # Logging the components
        self._add_loss_item('loss/kl_loss', kl_loss.item())
        self._add_loss_item('loss/node_loss', node_loss.item())
        self._add_loss_item('loss/graph_loss', graph_loss.item())
        self._add_loss_item('loss/beta', beta)

        # Computing the loss function
        loss = node_loss + graph_loss + beta * kl_loss
        return loss