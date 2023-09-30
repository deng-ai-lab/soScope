import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ..utils.modules import GraphEncoder, NodeDecoder, Nb_NodeDecoder, Poisson_NodeDecoder, JointNodeDecoder_Poisson_NB
from ..utils.modules import BiasSubGraphEncoder as EnhanceDecoder

from ..training.base import init_optimizer, Trainer
from ..utils.schedulers import ExponentialScheduler


def dot_product_decode(z):
    adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
    return adj_pred


################
# VGAE Trainer #
################
class VGAETrainer(Trainer):
    def __init__(self, gene_dim, sub_dim, z_dim,
                 optimizer_name='Adam',
                 lr=1e-4, beta_start_value=1e-3, beta_end_value=1,
                 scale=11.16418,
                 beta_n_iterations=100000, beta_start_iteration=50000,
                 head_num=1, **params):
        super(VGAETrainer, self).__init__(**params)

        self.gene_dim = gene_dim
        self.sub_dim = sub_dim
        self.z_dim = z_dim
        self.scale = scale
        self.head_num = head_num
        # Intialization of the encoder
        self.encoder_st = GraphEncoder(self.gene_dim, self.z_dim, head=self.head_num)
        self.decoder = EnhanceDecoder(self.z_dim, self.sub_dim, self.gene_dim)

        # Intialization of the decoder
        self.decoder_x = NodeDecoder(self.z_dim, self.gene_dim)

        # Adding the parameters of the estimator to the optimizer
        self.opt = init_optimizer(optimizer_name, [
            {'params': self.encoder_st.parameters(), 'lr': lr},
            {'params': self.decoder_x.parameters(), 'lr': lr},
        ])

        # Defining the prior distribution as a factorized normal distribution
        self.mu = nn.Parameter(torch.zeros(self.z_dim), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(self.z_dim), requires_grad=False)

        # prior for z
        self.z_prior = Normal(loc=self.mu, scale=self.sigma)
        self.z_prior = Independent(self.z_prior, 1)

        self.beta_scheduler = ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,
                                                   n_iterations=beta_n_iterations, start_iteration=beta_start_iteration)

    def _get_items_to_store(self):
        items_to_store = super(VGAETrainer, self)._get_items_to_store()

        # Add the mutual information estimator parameters to items_to_store
        items_to_store['encoder_st'] = self.encoder_st.state_dict()
        items_to_store['decoder_x'] = self.decoder_x.state_dict()
        return items_to_store

    def encode(self, x, edge_index):
        ##########
        # Trunk  #
        ##########

        # Encode a batch of data
        mu_z, sigma_z = self.encoder_st(x, edge_index)
        # print(mu_z, sigma_z)
        p_z_given_x_a = Independent(Normal(loc=mu_z, scale=sigma_z), 1)
        return p_z_given_x_a

    def decode(self, z, sigma_x):
        scale = sigma_x
        p_x_given_z = self.decoder_x(z, scale)
        return p_x_given_z

    def infer(self, st_data, sub_data, scale_factor=None, bias=False):
        self.eval()
        x_, edge_index = st_data[:2]
        x = x_[:, :self.gene_dim]
        subplot, sub_edge, sub_edge_value, index = sub_data
        sub_node_num = subplot.size()[-2]

        p_z_given_x_a = self.encode(x, edge_index)
        z = p_z_given_x_a.mean
        sigma_x = self.scale
        p_x_given_z = self.decode(z, sigma_x)
        x_hat = p_x_given_z.mean

        sub_x_pred = x_hat
        return sub_x_pred.detach()

    def estimate(self, st_data, sub_datas):
        device = self.get_device()

        for i, item in enumerate(st_data):
            st_data[i] = item.to(device)

        for sub_data in sub_datas:
            for j, item in enumerate(sub_data):
                if sub_data[j] is not None:
                    sub_data[j] = item.to(device)

        average_sq_error = 0
        count = 0
        for data in sub_datas:
            x_ = st_data[0]
            x = x_[:, :self.gene_dim]
            index = sub_data[-1]
            sub_x_pred = self.infer(st_data, data)
            average_sq_error = average_sq_error + torch.dist(sub_x_pred, x, p=2).item()
            count = count + 1
        average_sq_error = average_sq_error / count
        return average_sq_error

    def _train_step(self, st_data, sub_data, bias_data=None):
        loss = self._compute_loss(st_data)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def _compute_loss(self, st_data):
        x_, edge_index, edge_value = st_data
        x = x_[:, :self.gene_dim]
        spot_num = x.size()[0]

        p_z_given_x_a = self.encode(x, edge_index)
        # Sample from the posteriors with reparametrization
        z = p_z_given_x_a.rsample()
        sigma_x = self.scale

        # Decode
        prob_x_given_z = self.decode(z, sigma_x)

        # Distortion
        # the meaning is to minimize -Ep(z|x, A)[log p(x|z)]
        node_loss = -prob_x_given_z.log_prob(x.view(x.shape[0], -1))

        # KL Divergence
        # the meaning is to minimize -Ep(z|x,A)[log p(z|x,A) - log p(z)], that is KL[p(z|x,A)||p(z)]
        kl_loss = p_z_given_x_a.log_prob(z) - self.z_prior.log_prob(z)

        # Predict Adjacency Matrix
        adj_label = torch.sparse.FloatTensor(edge_index, edge_value, torch.Size([spot_num, spot_num]))
        adj_pred = dot_product_decode(z)
        graph_loss = F.binary_cross_entropy(adj_pred.view(-1), adj_label.to_dense().view(-1))

        # Average across the batch
        kl_loss = kl_loss.mean()
        node_loss = node_loss.mean()
        graph_loss = graph_loss.mean()

        # Update the value of beta according to the policy
        beta = self.beta_scheduler(self.iterations)

        # Logging the components
        self._add_loss_item('loss/kl_loss', kl_loss.item())
        self._add_loss_item('loss/node_loss', node_loss.item())
        self._add_loss_item('loss/graph_loss', graph_loss.item())
        self._add_loss_item('loss/beta', beta)

        # Computing the loss function
        loss = node_loss + graph_loss + beta * kl_loss
        return loss


class VGAETrainer_Poisson(VGAETrainer):
    def __init__(self, sub_node, **params):
        super(VGAETrainer_Poisson, self).__init__(**params)
        self.decoder_x = Poisson_NodeDecoder(self.z_dim, self.gene_dim)
        self.sub_node = sub_node

    def decode(self, z, sigma_x):
        scale = sigma_x
        p_x_given_z = self.decoder_x(z, scale)
        return p_x_given_z

    def _compute_loss(self, st_data):
        x_, edge_index, edge_value = st_data
        x = x_[:, :self.gene_dim]
        spot_num = x.size()[0]

        x_input = (x * self.sub_node)
        x_all = x_input.long()


        p_z_given_x_a = self.encode(x_input, edge_index)
        # Sample from the posteriors with reparametrization
        z = p_z_given_x_a.rsample()
        sigma_x = self.scale

        # Decode
        prob_x_given_z = self.decode(z, sigma_x)

        # Distortion
        # the meaning is to minimize -Ep(z|x, A)[log p(x|z)]
        node_loss = -prob_x_given_z.log_prob(x_all)
        # KL Divergence
        # the meaning is to minimize -Ep(z|x,A)[log p(z|x,A) - log p(z)], that is KL[p(z|x,A)||p(z)]
        kl_loss = p_z_given_x_a.log_prob(z) - self.z_prior.log_prob(z)

        # Predict Adjacency Matrix
        adj_label = torch.sparse.FloatTensor(edge_index, edge_value, torch.Size([spot_num, spot_num]))
        adj_pred = dot_product_decode(z)
        graph_loss = F.binary_cross_entropy(adj_pred.view(-1), adj_label.to_dense().view(-1))

        # Average across the batch
        kl_loss = kl_loss.mean()
        node_loss = node_loss.mean()
        graph_loss = graph_loss.mean()

        # Update the value of beta according to the policy
        beta = self.beta_scheduler(self.iterations)

        # Logging the components
        self._add_loss_item('loss/kl_loss', kl_loss.item())
        self._add_loss_item('loss/node_loss', node_loss.item())
        self._add_loss_item('loss/graph_loss', graph_loss.item())
        self._add_loss_item('loss/beta', beta)

        # Computing the loss function
        loss = node_loss + graph_loss + beta * kl_loss
        return loss


class VGAETrainer_NB(VGAETrainer):
    def __init__(self, sub_node, **params):
        super(VGAETrainer_NB, self).__init__(**params)
        self.decoder_x = Nb_NodeDecoder(self.z_dim, self.gene_dim)
        self.sub_node = sub_node

    def decode(self, z, sigma_x):
        scale = sigma_x
        p_x_given_z = self.decoder_x(z, scale)
        return p_x_given_z

    def _compute_loss(self, st_data):
        x_, edge_index, edge_value = st_data
        x = x_[:, :self.gene_dim]
        spot_num = x.size()[0]

        x_input = (x * self.sub_node)
        x_all = x_input.long()
        p_z_given_x_a = self.encode(x_input, edge_index)
        # Sample from the posteriors with reparametrization
        z = p_z_given_x_a.rsample()
        sigma_x = self.scale

        # Decode
        prob_x_given_z = self.decode(z, sigma_x)

        # Distortion
        # the meaning is to minimize -Ep(z|x, A)[log p(x|z)]
        node_loss = -prob_x_given_z.log_prob(x_all)
        # KL Divergence
        # the meaning is to minimize -Ep(z|x,A)[log p(z|x,A) - log p(z)], that is KL[p(z|x,A)||p(z)]
        kl_loss = p_z_given_x_a.log_prob(z) - self.z_prior.log_prob(z)

        # Predict Adjacency Matrix
        adj_label = torch.sparse.FloatTensor(edge_index, edge_value, torch.Size([spot_num, spot_num]))
        adj_pred = dot_product_decode(z)
        graph_loss = F.binary_cross_entropy(adj_pred.view(-1), adj_label.to_dense().view(-1))

        # Average across the batch
        kl_loss = kl_loss.mean()
        node_loss = node_loss.mean()
        graph_loss = graph_loss.mean()

        # Update the value of beta according to the policy
        beta = self.beta_scheduler(self.iterations)

        # Logging the components
        self._add_loss_item('loss/kl_loss', kl_loss.item())
        self._add_loss_item('loss/node_loss', node_loss.item())
        self._add_loss_item('loss/graph_loss', graph_loss.item())
        self._add_loss_item('loss/beta', beta)

        # Computing the loss function
        loss = node_loss + graph_loss + beta * kl_loss
        return loss


class VGAETrainer_Joint_PNB(VGAETrainer):
    def __init__(self, protein_dim, sub_node, **params):
        super(VGAETrainer_Joint_PNB, self).__init__(**params)
        self.protein_dim = protein_dim
        self.encoder_st = GraphEncoder(self.protein_dim + self.gene_dim, self.z_dim, head=self.head_num)
        self.decoder_x = JointNodeDecoder_Poisson_NB(self.z_dim, self.protein_dim, self.gene_dim)
        self.sub_node = sub_node

    def decode(self, z, scale):
        protein_given_z, gene_given_z = self.decoder_x(z, scale)
        return protein_given_z, gene_given_z

    def infer(self, st_data, sub_data, scale_factor=None, bias=False):
        self.eval()
        # Input data [X, A], [X_subplot, A_subplot], [index] index tells the position of subpolt in whole plot
        x_, edge_index = st_data[:2]
        subplot, sub_edge, sub_edge_value, index = sub_data

        # Calculate p(z|x,A), p(x|z) and p(e|x,z), and get X_hat for baseline
        # and X_hat/node + e for sub spot estimation.
        p_z_given_x_a = self.encode(x_, edge_index)
        z = p_z_given_x_a.mean
        sigma_x = self.scale
        protein_given_z, gene_given_z = self.decode(z, sigma_x)
        epsilon_p = protein_given_z.mean
        epsilon_g = gene_given_z.mean

        sub_x_pred = torch.cat([epsilon_p, epsilon_g], dim=1)
        return sub_x_pred.detach()

    def estimate(self, st_data, sub_datas):
        device = self.get_device()

        for i, item in enumerate(st_data):
            st_data[i] = item.to(device)

        for sub_data in sub_datas:
            for j, item in enumerate(sub_data):
                if sub_data[j] is not None:
                    sub_data[j] = item.to(device)

        average_sq_error = 0
        count = 0
        for data in sub_datas:
            x_ = st_data[0]
            index = sub_data[-1]
            sub_x_pred = self.infer(st_data, data)
            # sum_x_pred = torch.mean(sub_x_pred, dim=0)
            # average_sq_error = average_sq_error + torch.dist(sum_x_pred, x[index.item()], p=2).item()
            average_sq_error = average_sq_error + torch.dist(sub_x_pred, x_, p=2).item()
            count = count + 1
        average_sq_error = average_sq_error / count
        return average_sq_error

    def _compute_loss(self, st_data):
        # Input data [X, A], [X_subplot, A_subplot]
        x_, edge_index, edge_value = st_data
        spot_num = x_.size()[0]

        p_truth = x_[:, :self.protein_dim] * self.sub_node
        p_truth = p_truth.long()
        g_truth = x_[:, self.protein_dim:] * self.sub_node
        g_truth = g_truth.long()

        # Trunk encode
        p_z_given_x_a = self.encode(x_, edge_index)
        # Sample from the posteriors with reparametrization
        z = p_z_given_x_a.rsample()
        sigma_x = self.scale

        # Decode
        protein_given_z, gene_given_z = self.decode(z, sigma_x)

        # Distortion
        # the meaning is to minimize -Ep(z|x, A)[log p(x|z)]
        node_loss_p = -protein_given_z.log_prob(p_truth)
        node_loss_g = -gene_given_z.log_prob(g_truth)

        # KL Divergence
        # the meaning is to minimize -Ep(z|x,A)[log p(z|x,A) - log p(z)], that is KL[p(z|x,A)||p(z)]
        kl_loss = p_z_given_x_a.log_prob(z) - self.z_prior.log_prob(z)

        # Predict Adjacency Matrix
        adj_label = torch.sparse.FloatTensor(edge_index, edge_value, torch.Size([spot_num, spot_num]))
        adj_pred = dot_product_decode(z)
        graph_loss = F.binary_cross_entropy(adj_pred.view(-1), adj_label.to_dense().view(-1))

        # Average across the batch
        kl_loss = kl_loss.mean()
        node_loss = node_loss_p.mean() + node_loss_g.mean()
        graph_loss = graph_loss.mean()

        # Update the value of beta according to the policy
        beta = self.beta_scheduler(self.iterations)

        # Logging the components
        self._add_loss_item('loss/kl_loss', kl_loss.item())
        self._add_loss_item('loss/node_loss', node_loss.item())
        self._add_loss_item('loss/graph_loss', graph_loss.item())
        self._add_loss_item('loss/beta', beta)

        # Computing the loss function
        loss = node_loss + graph_loss + beta * kl_loss
        return loss