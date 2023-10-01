from typing import Union, Tuple, Optional

from torch_geometric.nn.inits import glorot
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


def add_self_loops_v2(edge_index, edge_weight: Optional[torch.Tensor] = None,
                      edge_attr: Optional[torch.Tensor] = None, edge_attr_reduce: str = "mean",
                      fill_value: float = 1., num_nodes: Optional[int] = None):
    r"""Extended method of torch_geometric.utils.add_self_loops that
    supports :attr:`edge_attr`."""
    N = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((N,), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    if edge_attr is not None:
        assert edge_attr.size(0) == edge_index.size(1)
        if edge_attr_reduce != "fill":
            loop_attr = scatter(edge_attr, edge_index[0], dim=0, dim_size=N,
                                reduce=edge_attr_reduce)
        else:
            loop_attr = edge_attr.new_full((N, edge_attr.size(1)), fill_value)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight, edge_attr


class GATEdgeConv(GATConv):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True,
                 edge_dim: int = None, edge_attr_reduce_for_self_loops: str = "mean",
                 edge_attr_fill_value: float = 1.,
                 **kwargs):

        assert edge_attr_reduce_for_self_loops in ["mean", "sum", "add", "mul", "min", "max", "fill"]
        self.edge_dim = edge_dim
        self.edge_attr_reduce_for_self_loops = edge_attr_reduce_for_self_loops
        self.edge_attr_fill_value = edge_attr_fill_value

        super().__init__(in_channels, out_channels, heads, concat,
                         negative_slope, dropout, add_self_loops, bias, **kwargs)

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.edge_dim is not None:
            self.lin_e = Linear(edge_dim, heads * out_channels, bias=False)
            self.att_e = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_e = None
            self.register_parameter('att_e', None)

        self._alpha = None
        self.reset_parameters_e()

    def reset_parameters_e(self):
        if self.edge_dim is not None:
            glorot(self.lin_e.weight)
            glorot(self.att_e)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        alpha_e: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, _, edge_attr = add_self_loops_v2(
                    edge_index, edge_attr=edge_attr,
                    edge_attr_reduce=self.edge_attr_reduce_for_self_loops,
                    fill_value=self.edge_attr_fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                assert edge_attr is None, \
                    "Using `edge_attr` not supported for SparseTensor `edge_index`."
                edge_index = set_diag(edge_index)

        if edge_attr is not None:
            edge_attr = self.lin_e(edge_attr).view(-1, H, C)
            alpha_e = (edge_attr * self.att_e).sum(dim=-1)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor, alpha_e: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), alpha_e=alpha_e,
                             size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                alpha_e: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = alpha if alpha_e is None else alpha + alpha_e
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


if __name__ == '__main__':
    REDUCE = "mean"
    _x = torch.randn((10, 7)).float()
    _edge_index = torch.arange(10).view(2, 5)
    _edge_attr = torch.randn((5, 17)).float()

    _layer = GATEdgeConv(7, 13, edge_dim=17, edge_attr_reduce_for_self_loops=REDUCE, heads=3, add_self_loops=True)
    print(_layer(_x, _edge_index, _edge_attr).size())  # torch.Size([10, 39])

    _layer = GATEdgeConv(7, 13, edge_dim=None, edge_attr_reduce_for_self_loops=REDUCE, heads=3, add_self_loops=True)
    print(_layer(_x, _edge_index).size())  # torch.Size([10, 39])

    _layer = GATEdgeConv(7, 13, edge_dim=17, edge_attr_reduce_for_self_loops=REDUCE, heads=3, add_self_loops=False)
    print(_edge_attr.size())
    print(_layer(_x, _edge_index, _edge_attr).size())  # torch.Size([10, 39])

    _layer = GATEdgeConv(7, 13, edge_dim=None, edge_attr_reduce_for_self_loops=REDUCE, heads=3, add_self_loops=False)
    print(_layer(_x, _edge_index).size())  # torch.Size([10, 39])