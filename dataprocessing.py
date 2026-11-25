"""
data preprocessing

For academic research purposes, several utility functions in this file have been refactored from the following projects:
1. https://github.com/melifluos/subgraph-sketching

"""


import os
import time

from math import inf
import random

import torch

import numpy as np

from torch_geometric.datasets import Amazon, Planetoid, Coauthor, Actor, WikipediaNetwork, WebKB
from ogb.linkproppred import PygLinkPropPredDataset

from torch_geometric.utils import add_self_loops, negative_sampling, to_undirected, degree

from torch_geometric.data import Data

from torch_geometric.transforms import RandomLinkSplit, ToSparseTensor

from torch_geometric.loader import LinkNeighborLoader

from torch_sparse import SparseTensor


def get_dataset(root, name: str, use_valedges_as_input=False, year=-1):
    if name.startswith('ogbl-'):
        dataset = PygLinkPropPredDataset(name=name, root=root)
        data = dataset[0]
        split_edge = dataset.get_edge_split()
        if name == 'ogbl-collab' and year > 0:  # filter out training edges before args.year
            data, split_edge = filter_by_year(data, split_edge, year)
        if 'edge_weight' in data:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
            # TEMP FIX: ogbl-collab has directed edges. adj_t.to_symmetric will
            # double the edge weight. temporary fix like this to avoid too dense graph.
            if name == "ogbl-collab":
                data.edge_weight = data.edge_weight/2
        # if 'edge' in split_edge['train']:
        #     key = 'edge'
        # else:
        #     key = 'source_node'
        # print("-"*20)
        # print(f"train: {split_edge['train'][key].shape[0]}")
        # print(f"{split_edge['train'][key]}")
        # print(f"valid: {split_edge['valid'][key].shape[0]}")
        # print(f"test: {split_edge['test'][key].shape[0]}")
        # print(f"max_degree:{degree(data.edge_index[0], data.num_nodes).max()}")
        data = ToSparseTensor(remove_edge_index=False)(data)
        data.adj_t = data.adj_t.to_symmetric()
        # Use training + validation edges for inference on test set.
        if use_valedges_as_input:
            val_edge_index = split_edge['valid']['edge'].t()
            full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
            data.full_adj_t = SparseTensor.from_edge_index(full_edge_index,
                                                    sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
            data.full_adj_t = data.full_adj_t.to_symmetric()
        else:
            data.full_adj_t = data.adj_t
        # make node feature as float
        if data.x is not None:
            data.x = data.x.to(torch.float)
        del data.edge_index

        return data, split_edge


# Helper utilities used by the two new strategies below
def _unique_undirected_edges(edge_index):
    if edge_index.numel() == 0:
        return edge_index.new_zeros((2, 0))
    src, dst = edge_index[0], edge_index[1]
    u = torch.minimum(src, dst)
    v = torch.maximum(src, dst)
    undirected = torch.stack([u, v], dim=0)
    unique = undirected.unique(dim=1)
    return unique


def _bidirectional_from_pairs(pairs, device, dtype):
    if pairs.numel() == 0:
        return torch.zeros((2, 0), device=device, dtype=dtype)
    a = pairs[0]
    b = pairs[1]
    src = torch.cat([a, b], dim=0)
    dst = torch.cat([b, a], dim=0)
    return torch.stack([src, dst], dim=0).to(device=device, dtype=dtype)


def generate_two_redundant_views_with_ratio(edge_index, shared_ratio=0.1, split_ratio=0.5):
    unique = _unique_undirected_edges(edge_index)
    device = edge_index.device
    dtype = edge_index.dtype
    n = unique.size(1)
    if n == 0:
        return [edge_index.new_zeros((2, 0)), edge_index.new_zeros((2, 0))]

    K = int(round(shared_ratio * n))
    perm = torch.randperm(n, device=device)
    shared_idx = perm[:K]
    remaining_idx = perm[K:]

    if remaining_idx.numel() > 0:
        r = torch.rand(remaining_idx.numel(), device=device)
        assign_to_0 = r < float(split_ratio)
        idx0 = remaining_idx[assign_to_0]
        idx1 = remaining_idx[~assign_to_0]
    else:
        idx0 = remaining_idx.new_empty((0,), dtype=torch.long)
        idx1 = remaining_idx.new_empty((0,), dtype=torch.long)

    # combine indices for each view
    if shared_idx.numel() + idx0.numel() > 0:
        view0_pairs = unique[:, torch.cat([shared_idx, idx0], dim=0)]
    else:
        view0_pairs = unique.new_zeros((2, 0))

    if shared_idx.numel() + idx1.numel() > 0:
        view1_pairs = unique[:, torch.cat([shared_idx, idx1], dim=0)]
    else:
        view1_pairs = unique.new_zeros((2, 0))

    view0 = _bidirectional_from_pairs(view0_pairs, device, dtype)
    view1 = _bidirectional_from_pairs(view1_pairs, device, dtype)
    return [view0, view1]


def generate_two_complementary_views(edge_index, p=0.5):

    edge_index_set = torch.arange(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
    edge_index_vector = torch.full_like(edge_index_set, p, dtype=torch.float32)
    mask = torch.bernoulli(edge_index_vector).to(torch.bool)

    view0 = edge_index[:, ~mask]
    view1 = edge_index[:, mask]

    def symmetrize(view):
        if view.numel() == 0:
            return view.new_zeros((2, 0))
        src, dst = view[0], view[1]
        rev = torch.stack([dst, src], dim=0)

        all_edges = torch.cat([view, rev], dim=1)
        unique_edges = torch.unique(all_edges, dim=1)
        return unique_edges

    view0 = symmetrize(view0)
    view1 = symmetrize(view1)

    return [view0, view1]


def generate_two_redundant_views(edge_index):
    p = 1/2
    edge_views = list()
    edge_index_set = torch.arange(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
    edge_index_vector = torch.full_like(edge_index_set, p, dtype=torch.float32)
    mask = torch.bernoulli(edge_index_vector).to(torch.bool)
    edge_views.append(edge_index[:, ~mask])
    edge_views.append(edge_index[:, mask])

    return edge_views

# def get_pos_neg_edges(data, sample_frac=1):
#     edge_index = data['edge_label_index']
#     labels = data['edge_label']
#     pos_edges = edge_index[:, labels == 1].t()
#     neg_edges = edge_index[:, labels == 0].t()
#     if sample_frac != 1:
#         n_pos = pos_edges.shape[0]
#         np.random.seed(123)
#         perm = np.random.permutation(n_pos)
#         perm = perm[:int(sample_frac * n_pos)]
#         pos_edges = pos_edges[perm, :]
#         neg_edges = neg_edges[perm, :]
#
#     return pos_edges, neg_edges

# def get_train_val_test_datasets(dataset_name,
#                                 splitting_data):
#
#     train_data, val_data, test_data = splitting_data['train'], splitting_data['valid'], splitting_data['test']
#
#     pos_train_edge, neg_train_edge = get_pos_neg_edges(train_data)
#     pos_val_edge, neg_val_edge = get_pos_neg_edges(val_data)
#     pos_test_edge, neg_test_edge = get_pos_neg_edges(test_data)
#
#     train_edge_data = {'pos_edge': pos_train_edge, 'neg_edge': neg_train_edge}
#     val_edge_data = {'pos_edge': pos_val_edge, 'neg_edge': neg_val_edge}
#     test_edge_data = {'pos_edge': pos_test_edge, 'neg_edge': neg_test_edge}
#
#     return train_data, val_data, test_data, train_edge_data, val_edge_data, test_edge_data


def get_loaders(data_set, batch_size,  num_neighbors_layers, num_workers=1):

    train_data, val_data, test_data = data_set['train'], data_set['valid'], data_set['test']

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=num_neighbors_layers,
        batch_size=batch_size,
        edge_label_index=train_data.edge_label_index,
        edge_label=train_data.edge_label,
        neg_sampling_ratio=0.0,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=num_neighbors_layers,
        batch_size=batch_size,
        edge_label_index=val_data.edge_label_index,
        edge_label=val_data.edge_label,
        neg_sampling_ratio=0.0,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=num_neighbors_layers,
        batch_size=batch_size,
        edge_label_index=test_data.edge_label_index,
        edge_label=test_data.edge_label,
        neg_sampling_ratio=0.0,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

def get_ogbl_loaders(dataset_name,
                data_set, split_edges,
                batch_size,  num_neighbors_layers, num_workers=1):

    train_data, val_data, test_data = data_set['train'], data_set['valid'], data_set['test']

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=num_neighbors_layers,
        batch_size=batch_size,
        edge_label_index=split_edges['train']['edge'].t(),
        edge_label=torch.ones(split_edges['train']['edge'].size(0), dtype=torch.float),
        neg_sampling_ratio=1.0, # 1:1 Dynamic negative sampling
        shuffle=True,
        num_workers=num_workers,
    )

    # as the val and test edges are often sampled they also need to be shuffled
    # the citation2 dataset has specific negatives for each positive and so can't be shuffled
    shuffle_val = False if dataset_name.startswith('ogbl-citation') else True
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=num_neighbors_layers,
        batch_size=batch_size,
        edge_label_index=split_edges['valid']['edge'].t(),
        edge_label=torch.ones(split_edges['valid']['edge'].size(0), dtype=torch.float),
        neg_sampling_ratio=0.0,
        shuffle=shuffle_val,
        num_workers=num_workers
    )
    shuffle_test = False if dataset_name.startswith('ogbl-citation') else True
    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=num_neighbors_layers,
        batch_size=batch_size,
        edge_label_index=split_edges['test']['edge'].t(),
        edge_label=torch.ones(split_edges['test']['edge'].size(0), dtype=torch.float),
        neg_sampling_ratio=0.0,
        shuffle=shuffle_test,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def load_ogbl_data(dataset_name, device, use_valedges_as_input=False, year=0):
    root = 'dataset'

    dataset = PygLinkPropPredDataset(name=dataset_name, root=root)
    if dataset_name == 'ogbl-ddi':
        dataset.data.x = torch.ones((dataset.data.num_nodes, 1))

    data = dataset[0]
    split_edges = dataset.get_edge_split()
    if dataset_name == 'ogbl-collab':
        if year > 0:
            # filter out training edges before args.year
            data, split_edges = filter_by_year(data, split_edges, year)

    if 'edge_weight' in data:
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
        # TEMP FIX: ogbl-collab has directed edges. adj_t.to_symmetric will
        # double the edge weight. temporary fix like this to avoid too dense graph.
        if dataset_name == "ogbl-collab":
            data.edge_weight = data.edge_weight/2

    data = ToSparseTensor(remove_edge_index=False)(data)
    data.adj_t = data.adj_t.to_symmetric()

    if use_valedges_as_input:
        val_edge_index = split_edges['valid']['edge'].t()
        full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index,
                                                       sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t

    if data.x is not None:
        data.x = data.x.to(torch.float)

    del data.edge_index
    data = data.to(device)

    return data, split_edges


# def load_ogbl_data(dataset_name, device, num_negs=1, year=0):
#     root = 'dataset'
#
#     dataset = PygLinkPropPredDataset(name=dataset_name, root=root)
#     if dataset_name == 'ogbl-ddi':
#         dataset.data.x = torch.ones((dataset.data.num_nodes, 1))
#
#     data_set = dataset[0]
#     split_edges = dataset.get_edge_split()
#     if dataset_name == 'ogbl-collab':
#         if year > 0:  # filter out training edges before args.year
#             data_set, split_edges = filter_by_year(data_set, split_edges, year)
#     dir_name = '_'.join(dataset_name.split('-'))
#     data_set = get_ogb_data(data_set, split_edges, dir_name, num_negs)
#
#     data_set['train'].to(device)
#     data_set['valid'].to(device)
#     data_set['test'].to(device)
#
#     return dataset, data_set, split_edges


def load_data(dataset_name, ratio, device):

    root = 'dataset'
    if dataset_name in {'Cora', 'Citeseer', 'Pubmed'}:
        dataset = Planetoid(root, dataset_name)

    elif dataset_name in {'Photo', 'Computers'}:
        dataset = Amazon(root, dataset_name)

    elif dataset_name in {'CS', 'Physics'}:
        dataset = Coauthor(root, dataset_name)

    elif dataset_name in {'Actor'}:
        dataset = Actor(root)

    elif dataset_name in {'Chameleon', 'Squirrel'}:
        dataset = WikipediaNetwork(root, dataset_name)

    elif dataset_name in {'Cornell', 'Texas', 'Wisconsin'}:
        dataset = WebKB(root, dataset_name)

    else:
        raise ValueError(dataset_name)

    # make random splits
    transform = RandomLinkSplit(num_val=ratio/2, num_test=ratio, is_undirected=True, add_negative_train_samples=True)
    train_data, val_data, test_data = transform(dataset.data)
    data_set = {'train': train_data, 'valid': val_data, 'test': test_data}

    data_set['train'].to(device)
    data_set['valid'].to(device)
    data_set['test'].to(device)

    return dataset, data_set


def filter_by_year(data, split_edge, year):
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= year).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()
    # create adjacency matrix
    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge


def get_same_source_negs(num_nodes, num_negs_per_pos, pos_edge):
    dst_neg = torch.randint(0, num_nodes, (1, pos_edge.size(1) * num_negs_per_pos), dtype=torch.long)
    src_neg = pos_edge[0].repeat_interleave(num_negs_per_pos)
    return torch.cat([src_neg.unsqueeze(0), dst_neg], dim=0)


def get_ogb_train_negs(split_edge, edge_index, num_nodes, num_negs=1, dataset_name=None):
    pos_edge = split_edge['train']['edge'].t()
    if dataset_name is not None and dataset_name.startswith('ogbl-citation'):
        neg_edge = get_same_source_negs(num_nodes, num_negs, pos_edge)
    else:  # any source is fine
        new_edge_index, _ = add_self_loops(edge_index)
        neg_edge = negative_sampling(
            new_edge_index, num_nodes=num_nodes,
            num_neg_samples=pos_edge.size(1) * num_negs)
    return neg_edge.t()

def make_obg_supervision_edges(split_edge, split, neg_edges=None):
    if neg_edges is not None:
        neg_edges = neg_edges
    else:
        neg_edges = split_edge[split]['edge_neg']

    pos_edges = split_edge[split]['edge']
    n_pos, n_neg = pos_edges.shape[0], neg_edges.shape[0]
    edge_label = torch.cat([torch.ones(n_pos), torch.zeros(n_neg)], dim=0)
    edge_label_index = torch.cat([pos_edges, neg_edges], dim=0).t()

    return edge_label, edge_label_index


def get_ogb_data(data, split_edges, dataset_name, num_negs=1):

    root = 'dataset'
    if num_negs == 1:
        negs_name = f'{root}/{dataset_name}/negative_samples.pt'
    else:
        negs_name = f'{root}/{dataset_name}/negative_samples_{num_negs}.pt'
    if os.path.exists(negs_name):
        train_negs = torch.load(negs_name)
    else:
        train_negs = get_ogb_train_negs(split_edges, data.edge_index, data.num_nodes, num_negs, dataset_name)
        torch.save(train_negs, negs_name)

    splits = {}
    for key in split_edges.keys():
        # the ogb datasets come with test and valid negatives, but you have to cook your own train negs
        neg_edges = train_negs if key == 'train' else None
        edge_label, edge_label_index = make_obg_supervision_edges(split_edges, key, neg_edges)
        # use the validation edges for message passing at test time
        # according to the rules https://ogb.stanford.edu/docs/leader_rules/ only collab can use val edges at test time
        if key == 'test' and dataset_name == 'ogbl-collab':
            vei = to_undirected(split_edges['valid']['edge'].t())
            edge_index = torch.cat([data.edge_index, vei], dim=1)
        else:
            edge_index = data.edge_index

        splits[key] = Data(x=data.x, edge_index=edge_index, edge_label=edge_label, edge_label_index=edge_label_index)

    return splits
