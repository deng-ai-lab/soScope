import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ..utils.modules import GraphEncoder, NodeDecoder
from ..utils.modules import BiasSubGraphEncoder as EnhanceDecoder
from ..training.base import init_optimizer, Trainer
from ..utils.schedulers import ExponentialScheduler



def dot_product_decode(z):
    adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
    return adj_pred


def block_merge(input_tensor, n):
    """
    Sum up every n rows of input
    Args:
        input_tensor: a tensor with dimension of [m*n,d]
        n: number of rows to merge

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
    output = torch.mm(sum_mat, input_tensor) / n
    return output


class soScope_Gaussian(Trainer):
    def __init__(self, gene_dim, sub_dim, z_dim, scale, sub_node=9,
                 optimizer_name='Adam',
                 lr=1e-4, weight_decay=0,
                 beta_start_value=1e-3, beta_end_value=1,
                 gamma_start_value=1, gamma_end_value=1e-3,
                 beta_n_iterations=100000, beta_start_iteration=50000,
                 graph_loss_weight=1, head_num=1, **params):
        super(soScope_Gaussian, self).__init__(**params)

        self.gene_dim = gene_dim
        self.sub_dim = sub_dim
        self.z_dim = z_dim
        self.head_num = head_num
        self.graph_loss_weight = graph_loss_weight
        self.scale = scale
        self.sub_node = sub_node

        # Intialization of the networks
        self.encoder_st = GraphEncoder(self.gene_dim, self.z_dim, head=self.head_num)
        self.decoder = EnhanceDecoder(self.z_dim, self.sub_dim, self.gene_dim)
        self.decoder_x = NodeDecoder(self.z_dim, self.gene_dim)
        # Adding the parameters of the estimator to the optimizer
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay

        self.opt = init_optimizer(optimizer_name, [
            {'params': self.encoder_st.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            {'params': self.decoder.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        ])

        # Defining the prior distribution as a factorized normal distribution
        self.mu = nn.Parameter(torch.zeros(self.z_dim), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(self.z_dim), requires_grad=False)

        # prior for z
        self.z_prior = Normal(loc=self.mu, scale=self.sigma)
        self.z_prior = Independent(self.z_prior, 1)
        self.beta_scheduler = ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,
                                                    n_iterations=beta_n_iterations,
                                                    start_iteration=beta_start_iteration)
        self.gamma_scheduler = ExponentialScheduler(start_value=gamma_start_value, end_value=gamma_end_value,
                                                   n_iterations=beta_n_iterations, start_iteration=beta_start_iteration)

    def _get_items_to_store(self):
        items_to_store = super(soScope_Gaussian, self)._get_items_to_store()
        items_to_store['encoder_st'] = self.encoder_st.state_dict()
        items_to_store['decoder'] = self.decoder.state_dict()
        return items_to_store

    def encode(self, x, edge_index):
        # Encode a batch of data
        mu_z, sigma_z = self.encoder_st(x, edge_index)
        p_z_given_x_a = Independent(Normal(loc=mu_z, scale=sigma_z), 1)
        return p_z_given_x_a

    def decode(self, sub_z, subplot):
        sub_node = self.sub_node
        sub_z_ = sub_z.repeat_interleave(sub_node, dim=0)
        sub_mu, sub_sigma = self.decoder(sub_z_, subplot)
        sub_sigma = torch.clamp(sub_sigma, self.scale/sub_node, 100 * self.scale / sub_node)
        p_epsilon_given_z_f = Independent(Normal(loc=sub_mu, scale=sub_sigma), 1)
        return p_epsilon_given_z_f

    def infer(self, st_data, sub_data, nonnegative=True, mode='mean'):
        assert mode == 'sum' or 'mean', 'No such mode, only sum and mean available'
        self.eval()
        # Input data [X, A], [Y, W]
        x_, edge_index = st_data[:2]
        x = x_[:, :self.gene_dim]

        subplot_, sub_edge, sub_edge_value, index = sub_data
        subplot = subplot_[:, :self.sub_dim]

        # Calculate p(z|x,A), and p(x_hat|x,z)
        p_z_given_x_a = self.encode(x, edge_index)
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
            mean_x_pred = block_merge(sub_x_pred, self.sub_node)
            average_sq_error = average_sq_error + torch.dist(mean_x_pred, x, p=2).item()
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
        # Input data [X, A], [X_subplot, A_subplot]
        x_, edge_index, edge_value = st_data
        x = x_[:, :self.gene_dim]

        spot_num = x.size()[-2]
        subplot_, sub_edge, sub_edge_value, index = sub_data
        subplot = subplot_[:, :self.sub_dim]

        subspot_num = subplot.size()[-2]

        p_z_given_x_a = self.encode(x, edge_index)

        # Sample from the posteriors with reparametrization
        z = p_z_given_x_a.rsample()
        # print(z)
        sub_z = z  # [1, z_dim]
        p_epsilon_given_z_f = self.decode(sub_z, subplot)
        epsilon = p_epsilon_given_z_f.rsample()  # [node_num, gene_dim]

        mse_loss = nn.MSELoss(reduction='mean')
        node_loss_ = (1/self.scale) * mse_loss((block_merge(epsilon, self.sub_node)).view(x.shape[0], -1),
                               x.view(x.shape[0], -1))

        node_loss = node_loss_.mean()
        # KL Divergence
        kl_loss_ = p_z_given_x_a.log_prob(z) - self.z_prior.log_prob(z)
        kl_loss = kl_loss_.mean()

        # Predict Adjacency Matrix
        sub_adj_pred = dot_product_decode(epsilon - x.repeat_interleave(self.sub_node, dim=0))
        sub_edge = torch.sparse.FloatTensor(sub_edge, sub_edge_value, torch.Size([subspot_num, subspot_num]))

        graph_loss_ = F.binary_cross_entropy(sub_adj_pred.view(-1), sub_edge.to_dense().view(-1))
        graph_loss = self.graph_loss_weight * graph_loss_.sum()

        beta = self.beta_scheduler(self.iterations)

        # Logging the components
        self._add_loss_item('loss/kl_loss', kl_loss.item())
        self._add_loss_item('loss/node_loss', node_loss.item())
        self._add_loss_item('loss/graph_loss', graph_loss.item())

        # Computing the loss function
        loss = node_loss + self.graph_loss_weight * graph_loss + beta * kl_loss
        return loss