import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from modules.utils import (normalize_laplacian_sparse, mean_average_distance_sparse, calculate_dirichlet, calculate_dirichlet_energy, add_self_loops)
from torch_geometric.nn import GATConv, GATv2Conv

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
        #if not laplacian.is_sparse:
        #    laplacian = laplacian.to_sparse()
        #print(f"GCNConv: adjacency shape: {adjacency.shape}, support shape: {support.shape}")
        output = torch.sparse.mm(adjacency, support)
        return output


class GCN(nn.Module):
    def __init__(self, in_features: int, hidden_dims: List[int], dropout: float = 0.):
        super(GCN, self).__init__()
        self.dropout = dropout
        # self.gcn_layers = nn.ModuleList()
        dims = [in_features] + hidden_dims
        gcn_layers = []
        for i in range(len(dims) - 1):
            gcn_layers.append(GCNConv(in_channels=dims[i], out_channels=dims[i + 1]))
        self.gcn_layers = nn.ModuleList(gcn_layers)


    def forward(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        for i, layer in enumerate(self.gcn_layers[:-1]):
            adj = adjacency[-(i + 1)] if isinstance(adjacency, list) else adjacency
            x = torch.relu(layer(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)

        adj = adjacency[0] if isinstance(adjacency, list) else adjacency
        logits = self.gcn_layers[-1](x, adj)
        logits = F.dropout(logits, p=self.dropout, training=self.training)

        memory_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
        
        return logits, memory_alloc
    
    def get_intermediate_outputs(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]) -> List[torch.Tensor]:
        intermediate_outputs = []
        for i, layer in enumerate(self.gcn_layers[:-1]):
            adj = adjacency[-(i + 1)] if isinstance(adjacency, list) else adjacency
            x = torch.relu(layer(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
            if 2**(i+1) in [2, 4, 8, 16, 32, 64, 128]:
                intermediate_outputs.append(x.clone())

        adj = adjacency[0] if isinstance(adjacency, list) else adjacency
        logits = self.gcn_layers[-1](x, adj)
        logits = F.dropout(logits, p=self.dropout, training=self.training)
        intermediate_outputs.append(logits)
        return intermediate_outputs
    
    def calculate_metrics(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]):
        adj = adjacency if not isinstance(adjacency, list) else adjacency.pop(0)
        energy1 = calculate_dirichlet(x, adj)
        energy2 = calculate_dirichlet_energy(x, adj)
        mad = mean_average_distance_sparse(x, adj)
        return energy1, energy2, mad
    
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
        dims = [in_features] + hidden_dims
        gcn_layers = []

        for i in range(len(hidden_dims)):
            gcn_layers.append(GCNConv(in_channels=dims[i], out_channels=dims[i + 1]))

        self.gcn_layers = nn.ModuleList(gcn_layers)


    def forward(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:

        for i, layer in enumerate(self.gcn_layers[:-1], start=1):
            adj = adjacency[-i] if isinstance(adjacency, list) else adjacency
            x_new = torch.relu(layer(x, adj))
            
            # residual connection
            x = x + x_new

            x = F.dropout(x, p=self.dropout, training=self.training)

        adj = adjacency[0] if isinstance(adjacency, list) else adjacency
        logits = self.gcn_layers[-1](x, adj)
        logits = F.dropout(logits, p=self.dropout, training=self.training)

        memory_alloc = torch.cuda.memory_allocated() / (1024 * 1024)

        return logits, memory_alloc
    


class GCNConvII(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, residual=False, variant=False):
        super(GCNConvII, self).__init__()
        self.variant = variant
        self.in_channels = in_channels * 2 if variant else in_channels
        self.out_channels = out_channels
        self.residual = residual
        self.weight = nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_channels)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / (l + 1) + 1)
        hi = torch.sparse.mm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], dim=1)
        else:
            support = (1 - alpha) * hi + alpha * h0
        output = theta * torch.mm(support, self.weight) + (1 - theta) * support
        if self.residual:
            output += input
        return output
    

class GCNII(nn.Module):
    def __init__(self, in_features: int, hidden_dims: List[int], nclass: int, dropout: float = 0.5, lamda: float = 0.5, alpha: float = 0.1, variant: bool = False):
        super(GCNII, self).__init__()
        self.dropout = dropout
        self.lamda = lamda
        self.alpha = alpha
        self.variant = variant
        
        dims = [in_features] + hidden_dims + [nclass]
        self.gcn_layers = nn.ModuleList()
        
        for i in range(len(dims) - 2):
            self.gcn_layers.append(GCNConvII(dims[i], dims[i + 1], residual=True, variant=variant))
        
        self.fc_in = nn.Linear(in_features, hidden_dims[0])
        self.fc_out = nn.Linear(hidden_dims[-1], nclass)
        self.act_fn = nn.ReLU()

    def forward(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Initial linear transformation
        x = F.dropout(x, self.dropout, training=self.training)
        h0 = self.act_fn(self.fc_in(x))
        
        # Store the initial transformed features
        _layers = [h0]
        layer_inner = h0
        
        for i, layer in enumerate(self.gcn_layers):
            adj = adjacency[-(i + 1)] if isinstance(adjacency, list) else adjacency
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(layer(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
        
        # Apply dropout and the final linear transformation
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        logits = self.fc_out(layer_inner)
        
        return F.log_softmax(logits, dim=1)


class GAT(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_dims: list[int]):
        super(GAT, self).__init__()

        dims = [in_features] + hidden_dims
        gat_layers = []
        for i in range(len(hidden_dims) - 1):
            gat_layers.append(GATConv(in_channels=dims[i],
                                      out_channels=dims[i + 1]))

        gat_layers.append(GATConv(in_channels=dims[-2], out_channels=dims[-1]))
        self.gat_layers = nn.ModuleList(gat_layers)

    def forward(self,
                x: torch.Tensor,
                edge_index: Union[torch.Tensor, list[torch.Tensor]],
                ) -> torch.Tensor:
        layerwise_adjacency = type(edge_index) == list

        for i, layer in enumerate(self.gat_layers[:-1], start=1):
            edges = edge_index[-i] if layerwise_adjacency else edge_index
            x = torch.relu(layer(x, edges))

        edges = edge_index[0] if layerwise_adjacency else edge_index
        logits = self.gat_layers[-1](x, edges)

        # memory
        memory_alloc = torch.cuda.memory_allocated() / (1024 * 1024)

        return logits, memory_alloc

# GATv2

class GATv2(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_dims: List[int],
                 heads: int = 1,
                 dropout: float = 0.6):
        super(GATv2, self).__init__()

        dims = [in_features] + hidden_dims
        gatv2_layers = []
        for i in range(len(hidden_dims) - 1):
            gatv2_layers.append(GATv2Conv(in_channels=dims[i],
                                          out_channels=dims[i + 1] // heads,
                                          heads=heads,
                                          dropout=dropout,
                                          concat=True))

        gatv2_layers.append(GATv2Conv(in_channels=dims[-2],
                                      out_channels=dims[-1],
                                      heads=heads,
                                      dropout=dropout,
                                      concat=False))
        self.gatv2_layers = nn.ModuleList(gatv2_layers)
        self.dropout = dropout

    def forward(self,
                x: torch.Tensor,
                edge_index: Union[torch.Tensor, List[torch.Tensor]],
                ) -> torch.Tensor:
        layerwise_adjacency = isinstance(edge_index, list)

        for i, layer in enumerate(self.gatv2_layers[:-1], start=1):
            edges = edge_index[-i] if layerwise_adjacency else edge_index
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(layer(x, edges))

        edges = edge_index[0] if layerwise_adjacency else edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.gatv2_layers[-1](x, edges)

        # memory
        memory_alloc = torch.cuda.memory_allocated() / (1024 * 1024)

        return logits, memory_alloc
    
    
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
    

    