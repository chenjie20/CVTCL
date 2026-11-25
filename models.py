import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
# from torch_scatter import scatter_add

from torch_geometric.utils import degree, remove_self_loops
from torch_sparse import SparseTensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree, remove_self_loops

try:
    from torch_sparse import SparseTensor
except ImportError:
    SparseTensor = None


class HeteroSeparatedConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        # 1. remove self-loops
        if SparseTensor is not None and isinstance(edge_index, SparseTensor):
            adj_t = edge_index.remove_diag()
            dst, src, _ = adj_t.coo()
        else:
            edge_index = edge_index.to(dtype=torch.long)
            edge_index, _ = remove_self_loops(edge_index)
            src, dst = edge_index[0], edge_index[1]

        # 2. Symmetric degree normalization (to prevent division by zero).
        device = dst.device() if callable(dst.device) else dst.device
        deg = torch.zeros(num_nodes, dtype=x.dtype, device=device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)

        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        # 3. Aggregation
        out = x.new_zeros((num_nodes, x.size(-1)))
        out.index_add_(0, dst, x[src] * norm.unsqueeze(-1))
        return out

class CVTCL(nn.Module):
    def __init__(
            self,
            edge_decoder,
            dim_feature,
            dim_hidden_feature,
            dropout=0.8
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(dim_feature, dim_hidden_feature))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
        self.edge_decoder = edge_decoder

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.edge_decoder is not None:
            self.edge_decoder.reset_parameters()

    def forward(self, x, adj):

        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, adj)
            x = self.activation(x)

        x = self.dropout(x)
        x = self.convs[-1](x, adj)
        x = self.activation(x)

        return x


class CVCTLv2(torch.nn.Module):
    def __init__(self, edge_decoder, in_channels, hidden_channels, out_channels, num_layers, dropout, num_cat_layers=1):
        super(CVCTLv2, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False, add_self_loops=False))
        # self.convs.append(HeteroSeparatedConv())
        # for _ in range(num_layers - 2):
        #     self.convs.append(HeteroSeparatedConv())
        # self.convs.append(HeteroSeparatedConv())
        self.edge_decoder = edge_decoder
        self.dropout = dropout
        self.num_cat_layers = num_cat_layers
        self.activation = nn.ELU()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.edge_decoder is not None:
            self.edge_decoder.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = self.activation (x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.num_cat_layers == 1:
            x = self.convs[-1](x, adj_t)
            x = self.activation(x)
        elif self.num_cat_layers == 2:
            h1 = self.activation(self.convs[0](x, adj_t))
            h1 = self.dropout(h1)
            h_all = torch.cat([x, h1], dim=-1)
            x = self.fusion(h_all)
        elif self.num_cat_layers == 3:
            h1 = self.activation(self.convs[0](x, adj_t))
            h1 = self.dropout(h1)
            h2 = self.activation(self.convs[1](h1, adj_t))
            h2 = self.dropout(h2)
            h_all = torch.cat([x, h1, h2], dim=-1)
            x = self.fusion(h_all)

        return x
