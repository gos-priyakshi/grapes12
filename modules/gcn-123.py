import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

class GCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GCNConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        output = torch.spmm(adjacency, support)
        return output

class GCN(nn.Module):
    def __init__(self, in_features: int, hidden_dims: List[int], dropout: float = 0.):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.gcn_layers = nn.ModuleList()
        self.enery_values = [] # List to store dirichlet energy values
        for i in range(len(hidden_dims) - 1):
            self.gcn_layers.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))

    def forward(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        self.enery_values = [] # Reset energy values
        for i, layer in enumerate(self.gcn_layers[:-1]):
            adj = adjacency[-i] if isinstance(adjacency, list) else adjacency
            x = torch.relu(layer(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Calculate dirichlet energy for each layer
            self.calculate_energy(x, adj)


        adj = adjacency[0] if isinstance(adjacency, list) else adjacency
        logits = self.gcn_layers[-1](x, adj)
        logits = F.dropout(logits, p=self.dropout, training=self.training)
        # Calculate dirichlet energy for the last layer
        self.calculate_energy(logits, adj)


        return logits
    
    def calculate_energy(self, x: torch.Tensor, adjacency: torch.Tensor):
        # Calculate Dirichlet energy
        laplacian = torch.spmm(adjacency, x) - x
        energy = torch.mean(torch.norm(laplacian, dim=1, p=2) ** 2)
        self.energy_values.append(energy)
        
    

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
    def __init__(self, in_channels: int, out_channels: int):
        super(GATConv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        output = torch.spmm(adjacency, support)
        return output
    

class GAT(nn.Module):
    def __init__(self, in_features: int, hidden_dims: List[int]):
        super(GAT, self).__init__()
        self.gat_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.gat_layers.append(GATConv(hidden_dims[i], hidden_dims[i + 1]))

    def forward(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        for i, layer in enumerate(self.gat_layers[:-1]):
            adj = adjacency[-i] if isinstance(adjacency, list) else adjacency
            x = torch.relu(layer(x, adj))

        adj = adjacency[0] if isinstance(adjacency, list) else adjacency
        logits = self.gat_layers[-1](x, adj)
        return logits
    
