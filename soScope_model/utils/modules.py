import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, NegativeBinomial, Poisson
from torch.nn.functional import softplus
from torch_geometric.nn import TransformerConv, Sequential

# DEFINE HIDDEN DIM
hidden_dim = 128

###################
#  ResNet block   #
###################
class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


##############################
# graph Encoder architecture #
##############################
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, z_dim, head):
        super(GraphEncoder, self).__init__()

        self.z_dim = z_dim
        dim_list = [input_dim, hidden_dim, z_dim]
        # Graph transformer
        self.net = Sequential('x, edge_index', [
            (TransformerConv(in_channels=dim_list[0], out_channels=dim_list[1], heads=head)
             , 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (TransformerConv(in_channels=dim_list[-2]*head, out_channels=dim_list[-2], heads=head),
             'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (TransformerConv(in_channels=dim_list[-2] * head, out_channels=dim_list[-1] * 2 // head, heads=head),
             'x, edge_index -> x')
        ])

    def forward(self, x, edge_index):

        params = self.net(x, edge_index)

        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive

        # return Independent(Normal(loc=mu, scale=sigma), 1)  # Return a factorized Normal distribution
        return mu, sigma


####################################
# Enhancement Decoder architecture #
####################################
class BiasSubGraphEncoder(nn.Module):
    def __init__(self, input_dim, sub_dim, z_dim):
        super(BiasSubGraphEncoder, self).__init__()

        self.z_dim = z_dim
        self.sub_dim = sub_dim
        dim_list = [input_dim + sub_dim, hidden_dim, z_dim]
        self.net = nn.Sequential(
            nn.Linear(dim_list[0], dim_list[1]),
            ResNet(
                nn.Sequential(
                    nn.Linear(dim_list[1], dim_list[1]),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim_list[1], dim_list[1]),
                    nn.ReLU(inplace=True),
                )
            ),
            ResNet(
                nn.Sequential(
                    nn.Linear(dim_list[1], dim_list[1]),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim_list[1], dim_list[1]),
                    nn.ReLU(inplace=True),
                )
            ),
            nn.Linear(dim_list[1], dim_list[-1]*2),
        )

    def forward(self, trunk_z_given_x, y):
        ########################
        # x|z,y ~ N(0,sigma^2) #
        ########################
        node_num = y.shape[0] // trunk_z_given_x.shape[0]
        input_feature = torch.cat([trunk_z_given_x / node_num, (2 - 1 / node_num) * y], dim=1)
        params = self.net(input_feature)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive
        return mu, sigma  # Return the sigma of a subplot

class JointSubGraphEncoder_Poisson_NB(nn.Module):
    def __init__(self, input_dim, sub_dim, protein_dim, gene_dim):
        super(JointSubGraphEncoder_Poisson_NB, self).__init__()
        self.protein_dim = protein_dim
        self.gene_dim = gene_dim
        self.z_dim = protein_dim + gene_dim * 2
        self.sub_dim = sub_dim
        dim_list = [input_dim + sub_dim, hidden_dim, self.z_dim]
        self.net = nn.Sequential(
            nn.Linear(dim_list[0], dim_list[1]),
            ResNet(
                nn.Sequential(
                    nn.Linear(dim_list[1], dim_list[1]),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim_list[1], dim_list[1]),
                    nn.ReLU(inplace=True),
                )
            ),
            ResNet(
                nn.Sequential(
                    nn.Linear(dim_list[1], dim_list[1]),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim_list[1], dim_list[1]),
                    nn.ReLU(inplace=True),
                )
            ),
            nn.Linear(dim_list[1], dim_list[-1]),
        )

    def forward(self, trunk_z_given_x, y):
        node_num = y.shape[0] // trunk_z_given_x.shape[0]
        input_feature = torch.cat([trunk_z_given_x / node_num, (2 - 1 / node_num) * y], dim=1)
        # params = self.net(input_feature, edge_index)
        params = self.net(input_feature)
        lambda_ = params[:, :self.protein_dim]
        lambda_ = softplus(lambda_) + 1e-7  # Make count always positive

        r = params[:, self.protein_dim: self.protein_dim+self.gene_dim]
        r = softplus(r) + 1e-7  # Make count always positive
        log_p = params[:, self.protein_dim+self.gene_dim: ]

        return lambda_, r, log_p


#############################
# node Decoder architecture #
#############################
class NodeDecoder(nn.Module):
    def __init__(self, z_dim, output_dim):
        super(NodeDecoder, self).__init__()

        self.z_dim = z_dim

        dim_list = [z_dim, hidden_dim, output_dim]
        # Graph transformer
        self.net = nn.Sequential(
            nn.Linear(dim_list[0], dim_list[1]),
            nn.ReLU(True),
            nn.Linear(dim_list[-2], dim_list[-2]),
            nn.ReLU(True),
            nn.Linear(dim_list[-2], dim_list[-1]),
        )

    def forward(self, z, scale):
        x = self.net(z)
        return Independent(Normal(loc=x, scale=scale), 1)

class Nb_NodeDecoder(nn.Module):
    def __init__(self, z_dim, output_dim):
        super(Nb_NodeDecoder, self).__init__()

        self.z_dim = z_dim
        self.output_dim = output_dim
        dim_list = [z_dim, hidden_dim, output_dim]
        # Graph transformer
        self.net = nn.Sequential(
            nn.Linear(dim_list[0], dim_list[1]),
            nn.ReLU(True),
            nn.Linear(dim_list[-2], dim_list[-2]),
            nn.ReLU(True),
            nn.Linear(dim_list[-2], dim_list[-1]*2),
        )

    def forward(self, z, scale=None):
        param = self.net(z)
        log_p = param[:, :self.output_dim]
        r = param[:, self.output_dim:]
        r = softplus(r) + 1e-7
        return Independent(NegativeBinomial(total_count=r, logits=log_p), 1)

class Poisson_NodeDecoder(nn.Module):
    def __init__(self, z_dim, output_dim):
        super(Poisson_NodeDecoder, self).__init__()

        self.z_dim = z_dim
        self.output_dim = output_dim
        dim_list = [z_dim, hidden_dim, output_dim]
        # Graph transformer
        self.net = nn.Sequential(
            nn.Linear(dim_list[0], dim_list[1]),
            nn.ReLU(True),
            nn.Linear(dim_list[-2], dim_list[-2]),
            nn.ReLU(True),
            nn.Linear(dim_list[-2], dim_list[-1]),
        )

    def forward(self, z, scale=None):
        param = self.net(z)
        r = softplus(param) + 1e-7
        return Independent(Poisson(r), 1)

class JointNodeDecoder_Poisson_NB(nn.Module):
    def __init__(self, z_dim, protein_dim, gene_dim):
        super(JointNodeDecoder_Poisson_NB, self).__init__()

        self.z_dim = z_dim
        self.protein_dim = protein_dim
        self.gene_dim = gene_dim
        self.output_dim = protein_dim + gene_dim * 2
        dim_list = [z_dim, hidden_dim, self.output_dim]
        # Graph transformer
        self.net = nn.Sequential(
            nn.Linear(dim_list[0], dim_list[1]),
            nn.ReLU(True),
            nn.Linear(dim_list[-2], dim_list[-2]),
            nn.ReLU(True),
            nn.Linear(dim_list[-2], dim_list[-1]),
        )

    def forward(self, z, scale):
        param = self.net(z)
        x = param[:, :self.protein_dim]
        x = softplus(x) + 1e-7

        r = param[:, self.protein_dim: self.protein_dim + self.gene_dim]
        r = softplus(r) + 1e-7
        log_p = param[:, self.protein_dim:self.protein_dim + self.gene_dim:]
        return Independent(Normal(loc=x, scale=scale), 1), Independent(NegativeBinomial(total_count=r, logits=log_p), 1)
