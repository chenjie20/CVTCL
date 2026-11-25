import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import torch
from torch_geometric.utils import negative_sampling, remove_self_loops
from torch_sparse import SparseTensor



class ExtendedEdgeDecoder(nn.Module):
    def __init__(self, dim, hidden=128, dropout=0.2):
        super().__init__()

        self.gate_logits = nn.Parameter(torch.zeros(dim))
        self.net = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def reset_parameters(self):
        nn.init.zeros_(self.gate_logits)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, z, edge_index, sigmoid=False):
        u = z[edge_index[0]]
        v = z[edge_index[1]]

        product = u * v
        abs_diff = torch.abs(u - v)

        gate = torch.sigmoid(self.gate_logits)
        gated_sum = gate * (u + v)

        ordered_diff = u - v

        edge_feat = torch.cat(
            [product, abs_diff, gated_sum, ordered_diff],
            dim=-1
        )

        logits = self.net(edge_feat).squeeze(-1)

        if sigmoid:
            return torch.sigmoid(logits)

        return logits


class SimpleEdgeDecoder(nn.Module):
    def __init__(self, dim, hidden=128, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, z, edge_index, sigmoid=False):
        u = z[edge_index[0]]
        v = z[edge_index[1]]

        # Part 1: element-wise product
        product = u * v

        logits = self.net(product).squeeze(-1)

        if sigmoid:
            return torch.sigmoid(logits)

        return logits


class EdgeDecoder(nn.Module):
    def __init__(self, dim, hidden=128, dropout=0.2):
        super().__init__()

        # Learnable feature-wise gate
        self.gate_logits = nn.Parameter(torch.zeros(dim))

        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def reset_parameters(self):
        nn.init.zeros_(self.gate_logits)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, z, edge_index, sigmoid=False):
        u = z[edge_index[0]]
        v = z[edge_index[1]]

        # Part 1: element-wise product
        product = u * v

        # Part 2: absolute difference
        abs_diff = torch.abs(u - v)

        # Part 3: gated sum
        gate = torch.sigmoid(self.gate_logits)
        gated_sum = gate * (u + v)

        edge_feat = torch.cat(
            [product, abs_diff, gated_sum], dim=-1
        )

        logits = self.net(edge_feat).squeeze(-1)

        if sigmoid:
            return torch.sigmoid(logits)

        return logits


class EdgeDecoderWithoutGate(nn.Module):
    def __init__(self, dim, hidden=128, dropout=0.2):
        super().__init__()

        # Learnable feature-wise gate
        self.gate_logits = nn.Parameter(torch.zeros(dim))

        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def reset_parameters(self):
        nn.init.zeros_(self.gate_logits)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, z, edge_index, sigmoid=False):
        u = z[edge_index[0]]
        v = z[edge_index[1]]

        # Part 1: element-wise product
        product = u * v

        # Part 2: absolute difference
        abs_diff = torch.abs(u - v)

        # Part 3: gated sum
        gated_sum = u + v

        edge_feat = torch.cat(
            [product, abs_diff, gated_sum], dim=-1
        )

        logits = self.net(edge_feat).squeeze(-1)

        if sigmoid:
            return torch.sigmoid(logits)

        return logits
