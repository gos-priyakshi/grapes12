import logging
import os
from typing import Dict, Tuple

import numpy as np
import psutil
import scipy.sparse as sp
import torch
from torch import Tensor
from torch.distributions import Bernoulli, Gumbel


def sample_neighborhoods_from_probs(logits: torch.Tensor,
                                    neighbor_nodes: torch.Tensor,
                                    num_samples: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Remove edges from an edge index, by removing nodes according to some
    probability.

    Uses Gumbel-max trick to sample from Bernoulli distribution. This is off-policy, since the original input
    distribution is a regular Bernoulli distribution.
    Args:
        logits: tensor of shape (N,), where N all the number of unique
            nodes in a batch, containing the probability of dropping the node.
        neighbor_nodes: tensor containing global node identifiers of the neighbors nodes
        num_samples: the number of samples to keep. If None, all edges are kept.
    """

    k = num_samples
    n = neighbor_nodes.shape[0]
    if k >= n:
        # TODO: Test this setting
        return neighbor_nodes, torch.sigmoid(
            logits.squeeze(-1)).log(), {}
    assert k < n
    assert k > 0

    b = Bernoulli(logits=logits.squeeze())

    # Gumbel-sort trick https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    gumbel = Gumbel(torch.tensor(0., device=logits.device), torch.tensor(1., device=logits.device))
    gumbel_noise = gumbel.sample((n,))
    perturbed_log_probs = b.probs.log() + gumbel_noise

    samples = torch.topk(perturbed_log_probs, k=k, dim=0, sorted=False)[1]

    # calculate the entropy in bits
    entropy = torch.tensor(-(b.probs * (b.probs).log2() + (1 - b.probs) * (1 - b.probs).log2()))

    min_prob = b.probs.min(-1)[0]
    max_prob = b.probs.max(-1)[0]

    if torch.isnan(entropy).any():
        nan_ind = torch.isnan(entropy)
        entropy[nan_ind] = 0.0

    std_entropy, mean_entropy = torch.std_mean(entropy)
    mask = torch.zeros_like(logits.squeeze(), dtype=torch.float)
    mask[samples] = 1

    neighbor_nodes = neighbor_nodes[mask.bool().cpu()]

    stats_dict = {"min_prob": min_prob,
                  "max_prob": max_prob,
                  "mean_entropy": mean_entropy,
                  "std_entropy": std_entropy}

    return neighbor_nodes, b.log_prob(mask), stats_dict


def get_neighborhoods(nodes: Tensor,
                      adjacency: sp.csr_matrix
                      ) -> Tensor:
    """Returns the neighbors of a set of nodes from a given adjacency matrix"""

    # Ensure nodes tensor is on the CPU
    nodes = nodes.cpu()
    
    neighborhood = adjacency[nodes].tocoo()
    neighborhoods = torch.stack([nodes[neighborhood.row],
                                 torch.tensor(neighborhood.col)],
                                dim=0)
    return neighborhoods


def slice_adjacency(adjacency: sp.csr_matrix, rows: Tensor, cols: Tensor):
    """Selects a block from a sparse adjacency matrix, given the row and column
    indices. The result is returned as an edge index.
    """
    #rows = rows.cpu() 
    row_slice = adjacency[rows]
    row_col_slice = row_slice[:, cols]
    slice = row_col_slice.tocoo()
    edge_index = torch.stack([rows[slice.row],
                              cols[slice.col]],
                             dim=0)
    return edge_index

def slice_adjacency_adj(adjacency: sp.csr_matrix, rows: Tensor, cols: Tensor):
    """Selects a block from a sparse adjacency matrix, given the row and column
    indices. The result is returned as a single adjacency matrix.
    """
    
    row_slice = adjacency[rows]
    row_col_slice = row_slice[:, cols]
    slice = row_col_slice.tocsr()

    adjacency_matrix = slice.todense()

    return adjacency_matrix

def convert_edge_index_to_adj(edge_index: Tensor, num_nodes: int): ### return a sparse adjacency instead
    """Converts an edge index to an adjacency matrix tensor"""
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    return adj

def convert_edge_index_to_adj_sparse(edge_index: Tensor, num_nodes: int): ### return a sparse adjacency instead
    """converts an edge index to a sparse adjacency matrix tensor"""
    # use torch.sparse_coo_tensor
    values = torch.ones(edge_index.size(1))

    # ensure edge_index and values are on the same device
    values = values.to(edge_index.device)

    # create the sparse adjacency matrix
    adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))

    return adj

def add_self_loops(adjacency):
    """Add self-loops to a sparse COO adjacency matrix."""
    
    num_nodes = adjacency.shape[0]

    # indices for the identity matrix
    self_loop_indices = torch.arange(num_nodes, device=adjacency.device)
    self_loop_indices = torch.stack([self_loop_indices, self_loop_indices], dim=0)

    # Values for the identity matrix 
    self_loop_values = torch.ones(num_nodes, device=adjacency.device)

    # Coalesce the adjacency tensor before getting its indices and values
    adjacency = adjacency.coalesce()

    # Concatenate self-loops with the original adjacency matrix
    new_indices = torch.cat([adjacency.indices(), self_loop_indices], dim=1)
    new_values = torch.cat([adjacency.values(), self_loop_values])

    # Create the new sparse adjacency matrix with self-loops
    new_adjacency = torch.sparse_coo_tensor(new_indices, new_values, adjacency.size(), device=adjacency.device)
    new_adjacency = new_adjacency.coalesce() 

    return new_adjacency

def normalize_laplacian(adjacency: Tensor):
    """Computes the normalized graph Laplacian of adjacency matrix."""
    rowsum = torch.sum(adjacency, dim=1)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    laplacian = adjacency.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)
    return laplacian

def normalize_laplacian_sparse(adjacency: torch.sparse.FloatTensor):
    """Computes the normalized graph Laplacian of adjacency matrix."""
    rowsum = torch.sparse.sum(adjacency, dim=1).to_dense()
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.

    indices = torch.arange(len(d_inv_sqrt), device=adjacency.device)
    indices = torch.stack([indices, indices], dim=0)
    d_inv_sqrt_sparse = torch.sparse.FloatTensor(indices, d_inv_sqrt, adjacency.size())

    # compute D^(-1/2) * A * D^(-1/2)
    d_inv_sqrt_mat = torch.sparse.mm(d_inv_sqrt_sparse, adjacency)
    laplacian = torch.sparse.mm(d_inv_sqrt_mat, d_inv_sqrt_sparse)

    return laplacian

def calculate_dirichlet_energy(x : Tensor, adj: torch.sparse.FloatTensor):
    """Calculates the Dirichlet energy of node features x on a graph with adjacency matrix adj."""
    num_nodes: int = x.shape[0]
    de: Tensor = 0

    def inner(x_i: Tensor, x_js: Tensor) -> Tensor:
        return torch.norm(x_i - x_js, ord=2, dim=1).pow(2).sum()

    for node_index in range(num_nodes):
        own_feat_vector = x[[node_index], :]
        nbh_indices = torch.nonzero(adj[node_index].to_dense(), as_tuple=True)[0]
        nbh_feat_matrix = x[nbh_indices, :]

        de += inner(own_feat_vector, nbh_feat_matrix)

    return torch.sqrt(de / num_nodes).item()

def calculate_dirichlet_energy_sparse(x : Tensor, adj: torch.sparse.FloatTensor):
    """Calculates the Dirichlet energy of node features x on a graph with adjacency matrix adj."""
    # add self-loops
    adj = add_self_loops(adj)
    # calculate the laplacian
    laplacian = normalize_laplacian_sparse(adj)
    
    # Create a sparse identity matrix
    indices = torch.arange(adj.size(0), device=adj.device)
    indices = torch.stack([indices, indices], dim=0)
    values = torch.ones(adj.size(0), device=adj.device)
    eye_sparse = torch.sparse_coo_tensor(indices, values, (adj.size(0), adj.size(0)))

    # augmented normalized laplacian 
    laplacian = laplacian - eye_sparse
    laplacian = laplacian.to(x.device)
    # calculate the Dirichlet energy
    energy = torch.mm(x.t(), laplacian.to_dense()).mm(x)
    # trace 
    energy = torch.trace(energy)

    return energy.item()


def cosine_similarity(x: Tensor, y: Tensor):
    """Calculates the cosine similarity between two tensors x and y."""
    x = x / x.norm(dim=1)[:, None]
    y = y / y.norm(dim=1)[:, None]
    return torch.mm(x, y.t())

def mean_average_distance_sparse(x: Tensor, adj: torch.sparse.FloatTensor):
    """calculates the mean average distance of node features x """

    num_nodes = x.size(0)
    mad = 0.0
    valid_pairs = 0

    for i in range(num_nodes):
        # coalesce the adjacency matrix
        adj = adj.coalesce()
        # get the neighbors of node i
        neighbours = adj._indices()[1, adj._indices()[0] == i]
        num_neighbours = neighbours.size(0)

        if num_neighbours == 0:
            continue

        valid_pairs += 1

        for j in neighbours:
            similarity = torch.cosine_similarity(x[i], x[j], dim=0)
            mad += 1 - similarity

    mad = mad / valid_pairs if valid_pairs > 0 else 0.0
    return mad


def mean_average_distance(x: Tensor, adj: Tensor):
    """calculates the mean average distance of node features x """

    num_nodes = x.size(0)
    mad = 0.0
    valid_pairs = 0

    for i in range(num_nodes):
        # get the neighbors of node i
        neighbours = adj[i].nonzero(as_tuple=False).flatten()
        num_neighbours = neighbours.size(0)

        if num_neighbours == 0:
            continue

        valid_pairs += 1

        for j in neighbours:
            similarity = torch.cosine_similarity(x[i], x[j], dim=0)
            mad += 1 - similarity

    mad = mad / valid_pairs if valid_pairs > 0 else 0.0
    return mad


class TensorMap:
    """A class used to quickly map integers in a tensor to an interval of
    integers from 0 to len(tensor) - 1. This is useful for global to local
    conversions.

    Example:
        >>> nodes = torch.tensor([22, 32, 42, 52])
        >>> node_map = TensorMap(size=nodes.max() + 1)
        >>> node_map.update(nodes)
        >>> node_map.map(torch.tensor([52, 42, 32, 22, 22]))
        tensor([3, 2, 1, 0, 0])
    """

    def __init__(self, size):
        self.map_tensor = torch.empty(size, dtype=torch.long)
        self.values = torch.arange(size)

    def update(self, keys: Tensor):
        values = self.values[:len(keys)]
        self.map_tensor[keys] = values

    def map(self, keys):
        return self.map_tensor[keys]


def get_logger():
    """Get a default logger that includes a timestamp."""
    logger = logging.getLogger('')
    logger.handlers = []
    ch = logging.StreamHandler()
    str_fmt = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    formatter = logging.Formatter(str_fmt, datefmt='%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('INFO')

    return logger


def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx


# From PyGAS, PyTorch Geometric Auto-Scale: https://github.com/rusty1s/pyg_autoscale/tree/master
def index2mask(idx: Tensor, size: int) -> Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask


def gen_masks(y: Tensor, train_per_class: int = 20, val_per_class: int = 30,
              num_splits: int = 20) -> Tuple[Tensor, Tensor, Tensor]:
    num_classes = int(y.max()) + 1

    train_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), num_splits, dtype=torch.bool)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        perm = torch.stack(
            [torch.randperm(idx.size(0)) for _ in range(num_splits)], dim=1)
        idx = idx[perm]

        train_idx = idx[:train_per_class]
        train_mask.scatter_(0, train_idx, True)
        val_idx = idx[train_per_class:train_per_class + val_per_class]
        val_mask.scatter_(0, val_idx, True)

    test_mask = ~(train_mask | val_mask)

    return train_mask, val_mask, test_mask


# Function to return memory usage in MB
def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)  # Convert bytes to MB

