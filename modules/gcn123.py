import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from modules.utils import (normalize_laplacian_sparse, mean_average_distance_sparse, calculate_dirichlet_energy, add_self_loops)

class GCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GCNConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        #print(f"GCNConv: x shape: {x.shape}, weight shape: {self.weight.shape}")
        support = torch.mm(x, self.weight)

        # add self-loops to sparse coo tensor and normalize the adjacency matrix
        adjacency = add_self_loops(adjacency)
        adjacency = normalize_laplacian_sparse(adjacency)

         # Move adjacency to the same device as x
         # check device of x
        # print(f"GCNConv: x device: {x.device}, adjacency device: {adjacency.device}")
        adjacency = adjacency.to(x.device)
        if not adjacency.is_sparse:
            adjacency = adjacency.to_sparse()
        #print(f"GCNConv: adjacency shape: {adjacency.shape}, support shape: {support.shape}")
        output = torch.spmm(adjacency, support)
        return output


class GCN(nn.Module):
    def __init__(self, in_features: int, hidden_dims: List[int], dropout: float = 0.):
        super(GCN, self).__init__()
        self.dropout = dropout
        # self.gcn_layers = nn.ModuleList()
        #self.energy_values = []  # List to store Dirichlet energy values
        #self.mad_values = []  # List to store MAD values
        dims = [in_features] + hidden_dims
        gcn_layers = []
        for i in range(len(dims) - 1):
            gcn_layers.append(GCNConv(in_channels=dims[i], out_channels=dims[i + 1]))
        self.gcn_layers = nn.ModuleList(gcn_layers)


    def forward(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        #self.energy_values = [] # Reset energy values
        #self.mad_values = [] # Reset mean average distance values
        #print(f"GCN: initial x shape: {x.shape}")
        for i, layer in enumerate(self.gcn_layers[:-1]):
            adj = adjacency[-i] if isinstance(adjacency, list) else adjacency
            x = torch.relu(layer(x, adj))
            #print(f"GCN: after layer {i}, x shape: {x.shape}")
            # Calculate dirichlet energy for each layer
            #self.calculate_and_store_metrics(x, adj)
            x = F.dropout(x, p=self.dropout, training=self.training)
            

        adj = adjacency[0] if isinstance(adjacency, list) else adjacency
        logits = self.gcn_layers[-1](x, adj)
        #print(f"GCN: final logits shape: {logits.shape}")
        # Calculate dirichlet energy for the last layer
        #self.calculate_and_store_metrics(logits, adj)
        logits = F.dropout(logits, p=self.dropout, training=self.training)

        memory_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
        
        return logits, memory_alloc
    
    def calculate_metrics(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]):
        adj = adjacency[0] if isinstance(adjacency, list) else adjacency

        energy = calculate_dirichlet_energy(x, adj)
        mad = mean_average_distance_sparse(x, adj)
        return energy, mad
    
    #def calculate_and_store_metrics(self, x: torch.Tensor, adj: torch.Tensor):
     #   energy = calculate_dirichlet_energy_sparse(x, adj)
     #   mad = mean_average_distance_sparse(x, adj)
     #   self.energy_values.append(energy)
     #   self.mad_values.append(mad)

    #def get_dirichlet_energy(self):
    #    """Returns the list of Dirichlet energy values at each layer after training."""
    #    return self.energy_values
    
    #def get_mean_average_distance(self):
    #    """Returns the list of mean average distance values at each layer after training."""
    #    return self.mad_values

        
    

class ResGCN(nn.Module):
    def __init__(self, in_features: int, hidden_dims: List[int], dropout: float = 0.):
        super(ResGCN, self).__init__()
        self.dropout = dropout
        self.gcn_layers = nn.ModuleList()
        self.residual_transforms = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.gcn_layers.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))
            if hidden_dims[i] != hidden_dims[i + 1]:
                self.residual_transforms.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            else:
                self.residual_transforms.append(None)

    def forward(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        for i, (layer, residual_transform) in enumerate(zip(self.gcn_layers, self.residual_transforms)):
            adj = adjacency[-i] if isinstance(adjacency, list) else adjacency
            x_res = x
            x = torch.relu(layer(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
            if residual_transform is not None:
                x_res = residual_transform(x_res)
            x = x + x_res
        return x
    

class GATConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, dropout: float = 0.0):
        super(GATConv, self).__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.attention = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attention)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        H, C = self.heads, self.out_channels
        support = torch.mm(x, self.weight).view(-1, H, C)
        indices = adjacency._indices()
        N = x.size(0)

        edge_h = torch.cat((support[indices[0, :], :, :], support[indices[1, :], :, :]), dim=-1)
        edge_e = torch.exp(self.leakyrelu((edge_h * self.attention).sum(dim=-1, keepdim=True)))

        e_rowsum = torch.sparse.sum(adjacency, dim=1).to_dense().view(-1, 1, H)
        edge_e /= e_rowsum[indices[0, :], :, :]

        edge_e = torch.sparse_coo_tensor(indices, edge_e.squeeze(-1), torch.Size([N, N, H]))
        edge_e = edge_e.to_dense()

        support = support * edge_e.unsqueeze(-1)
        output = support.sum(dim=1)

        if self.dropout:
            output = F.dropout(output, p=self.dropout, training=self.training)

        return output


class GAT(nn.Module):
    def __init__(self, in_features: int, hidden_dims: list[int], heads: list[int], dropout: float = 0.0):
        super(GAT, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.dropout = dropout
        dims = [in_features] + hidden_dims
        for i in range(len(dims) - 1):
            self.gat_layers.append(GATConv(dims[i], dims[i + 1], heads[i], dropout))

    def forward(self, x: torch.Tensor, adjacencies: list[torch.Tensor]) -> torch.Tensor:
        for i, (layer, adj) in enumerate(zip(self.gat_layers, adjacencies)):
            x = F.relu(layer(x, adj))
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# simple attention in graph neural network


# graphSAGE implementation

class GraphSAGELayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, aggregator_type: str = 'mean', dropout: float = 0.):
        super(GraphSAGELayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregator_type = aggregator_type
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)
        self.dropout = dropout

        if aggregator_type == 'pool':
            self.pool = nn.Linear(in_channels, in_channels)
            nn.init.xavier_uniform_(self.pool.weight)
        elif aggregator_type == 'lstm':
            self.lstm = nn.LSTM(in_channels, in_channels, batch_first=True)

    def forward(self, x: torch.Tensor, adj_list: list) -> torch.Tensor:
        neighbors = [torch.index_select(x, 0, indices) for indices in adj_list]
        if self.aggregator_type == 'mean':
            agg_neighbors = torch.stack(neighbors).mean(dim=0)
        elif self.aggregator_type == 'pool':
            pooled_neighbors = F.relu(self.pool(torch.stack(neighbors)))
            agg_neighbors = pooled_neighbors.max(dim=0)[0]
        elif self.aggregator_type == 'lstm':
            lstm_out, _ = self.lstm(torch.stack(neighbors))
            agg_neighbors = lstm_out[:, -1, :]

        # Concatenate self feature and aggregated neighbors' features
        h = torch.cat([x, agg_neighbors], dim=1)
        h = torch.mm(h, self.weight)

        if self.dropout:
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

class GraphSAGE(nn.Module):
    def __init__(self, in_features: int, hidden_dims: list, num_classes: int, aggregator_type: str = 'mean', dropout: float = 0.):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.energy_values = []  # List to store Dirichlet energy values
        dims = [in_features] + hidden_dims + [num_classes]
        for i in range(len(dims) - 1):
            self.layers.append(GraphSAGELayer(dims[i], dims[i + 1], aggregator_type, dropout))

    def forward(self, x: torch.Tensor, adj_lists: list) -> torch.Tensor:
        self.energy_values = []  # Reset energy values
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, adj_lists))
            # Calculate Dirichlet energy for each layer
            self.calculate_energy(x, adj_lists)
        logits = self.layers[-1](x, adj_lists)
        logits = F.dropout(logits, p=self.dropout, training=self.training)
        # Calculate Dirichlet energy for the last layer
        self.calculate_energy(logits, adj_lists)
        return logits

    def calculate_energy(self, x: torch.Tensor, adj_lists: list):
        # Here, we compute Dirichlet energy using adjacency lists
        laplacian = torch.stack([torch.index_select(x, 0, indices).mean(dim=0) - x for indices in adj_lists])
        energy = torch.mean(torch.norm(laplacian, dim=1, p=2) ** 2)
        self.energy_values.append(energy)
    

# GatedGCN implementation

class GatedGCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GatedGCNConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.gate_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.gate_weight)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        gate = torch.mm(x, self.gate_weight)
        output = torch.spmm(adjacency, support)
        output = torch.sigmoid(torch.spmm(adjacency, gate)) * output
        return output
    
class GatedGCN(nn.Module):
    def __init__(self, in_features: int, hidden_dims: List[int]):
        super(GatedGCN, self).__init__()
        self.gcn_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.gcn_layers.append(GatedGCNConv(hidden_dims[i], hidden_dims[i + 1]))

    def forward(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        for i, layer in enumerate(self.gcn_layers[:-1]):
            adj = adjacency[-i] if isinstance(adjacency, list) else adjacency
            x = torch.relu(layer(x, adj))

        adj = adjacency[0] if isinstance(adjacency, list) else adjacency
        logits = self.gcn_layers[-1](x, adj)
        return logits
    

    