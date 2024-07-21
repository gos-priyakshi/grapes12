import math
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

        target_layers = [2, 4, 8, 16, 32, 64, 128]
        
        for i, layer in enumerate(self.gcn_layers[:-1]):
            adj = adjacency[-(i + 1)] if isinstance(adjacency, list) else adjacency
            x = torch.relu(layer(x, adj))
            x = F.dropout(x, p=self.dropout)
            
            # store if current layer is in target_layers
            if i + 1 in target_layers:
                print(f"Storing intermediate output for layer {i + 1}")
                intermediate_outputs.append(x.clone())

        adj = adjacency[0] if isinstance(adjacency, list) else adjacency
        logits = self.gcn_layers[-1](x, adj)
        logits = F.dropout(logits, p=self.dropout)
        intermediate_outputs.append(logits)
        return intermediate_outputs
    
    def calculate_metrics(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]):
        adj = adjacency if not isinstance(adjacency, list) else adjacency.pop(0)
        energy1 = calculate_dirichlet(x, adj)
        energy2 = calculate_dirichlet_energy(x, adj)
        #mad = mean_average_distance_sparse(x, adj)
        return energy1, energy2
    
        
#class ResGCN(nn.Module):
#    def __init__(self, in_features: int, hidden_dims: List[int], dropout: float = 0.):
#        super(ResGCN, self).__init__()
#        self.dropout = dropout
#        dims = [in_features] + hidden_dims
#        gcn_layers = []
#        residual_transforms = []

#        for i in range(len(hidden_dims)):
#            gcn_layers.append(GCNConv(in_channels=dims[i], out_channels=dims[i + 1]))
#            if dims[i] != dims[i + 1]:
#                residual_transforms.append(nn.Linear(dims[i], dims[i + 1]))
#            else:
#                residual_transforms.append(nn.Identity())

#        self.gcn_layers = nn.ModuleList(gcn_layers)
#        self.residual_transforms = nn.ModuleList(residual_transforms)
        

#    def forward(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:

#        for i, (layer, transform) in enumerate(zip(self.gcn_layers[:-1], self.residual_transforms), start=1):
#            adj = adjacency[-i] if isinstance(adjacency, list) else adjacency
#            x_new = torch.relu(layer(x, adj)) # F(X_{n-1}, G)
#            x_res = transform(x) 
            # residual connection
#            x = x_res + x_new # X_n = X_{n-1} + F(X_{n-1}, G)
#            x = F.dropout(x, p=self.dropout, training=self.training)

#        adj = adjacency[0] if isinstance(adjacency, list) else adjacency
#        logits = self.gcn_layers[-1](x, adj)
#        logits = F.dropout(logits, p=self.dropout, training=self.training)

#        memory_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
#        return logits, memory_alloc
    
#    def get_intermediate_outputs(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]) -> List[torch.Tensor]:
#        intermediate_outputs = []
#        target_layers = [2, 4, 8, 16, 32, 64, 128]

#        for i, (layer, transform) in enumerate(zip(self.gcn_layers[:-1], self.residual_transforms[:-1]), start=1):
#            adj = adjacency[-i] if isinstance(adjacency, list) else adjacency
#            x_new = torch.relu(layer(x, adj))
#            x_res = transform(x)
#            x = x_res + x_new
#           x = F.dropout(x, p=self.dropout)
#            if i in target_layers:
#                intermediate_outputs.append(x.clone())
#
#        adj = adjacency[0] if isinstance(adjacency, list) else adjacency
#        logits = self.gcn_layers[-1](x, adj)
#        logits = F.dropout(logits, p=self.dropout)
#        intermediate_outputs.append(logits)
#        return intermediate_outputs
#
#    def calculate_metrics(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]):
#        adj = adjacency if not isinstance(adjacency, list) else adjacency.pop(0)
#        energy1 = calculate_dirichlet(x, adj)
#        energy2 = calculate_dirichlet_energy(x, adj)
#        return energy1, energy2
    

class ResGCN(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_dims: list[int], dropout: float=0.):
        super(ResGCN, self).__init__()
        self.dropout = dropout
        dims = [in_features] + hidden_dims
        gcn_layers = []
        self.transform = nn.Linear(in_features, hidden_dims[0])  # Transform input features to hidden dimension
        for i in range(len(hidden_dims)):
            gcn_layers.append(GCNConv(in_channels=dims[i], out_channels=dims[i + 1]))
        self.gcn_layers = nn.ModuleList(gcn_layers)

    def forward(self, x: torch.Tensor, adjacency: Union[torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:

        x_0 = self.transform(x)

        for i, layer in enumerate(self.gcn_layers[:-1]):
            adj = adjacency[-(i + 1)] if isinstance(adjacency, list) else adjacency
            x = torch.relu(layer(x, adj)) + x_0  # skip connection here
            x = F.dropout(x, p=self.dropout, training=self.training)

        adj = adjacency[0] if isinstance(adjacency, list) else adjacency
        logits = self.gcn_layers[-1](x, adj)
        logits = F.dropout(logits, p=self.dropout, training=self.training)

        memory_alloc = torch.cuda.memory_allocated() / (1024 * 1024)

        return logits, memory_alloc
    
    def get_intermediate_outputs(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]) -> List[torch.Tensor]:

        intermediate_outputs = []
        target_layers = [2, 4, 8, 16, 32, 64, 128]

        x_0 = self.transform(x)

        for i, layer in enumerate(self.gcn_layers[:-1]):
            adj = adjacency[-(i + 1)] if isinstance(adjacency, list) else adjacency
            x = torch.relu(layer(x, adj)) + x_0
            x = F.dropout(x, p=self.dropout)
            if i + 1 in target_layers:
                intermediate_outputs.append(x.clone())

        adj = adjacency[0] if isinstance(adjacency, list) else adjacency
        logits = self.gcn_layers[-1](x, adj)
        logits = F.dropout(logits, p=self.dropout)
        intermediate_outputs.append(logits)

        return intermediate_outputs
    
    def calculate_metrics(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]):
        adj = adjacency if not isinstance(adjacency, list) else adjacency.pop(0)
        energy1 = calculate_dirichlet(x, adj)
        energy2 = calculate_dirichlet_energy(x, adj)
        return energy1, energy2
    


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
        # initialize the weights
        stdv = 1. / math.sqrt(self.out_channels)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):

        # add self-loops to sparse coo tensor and normalize the adjacency matrix
        adj = add_self_loops(adj)
        adj = normalize_laplacian_sparse(adj)

        # calculate theta based on lambda and l
        theta = math.log(lamda/l + 1)
        print(f"shapes: input: {input.shape}, adj: {adj.shape}, h0: {h0.shape}, weight: {self.weight.shape}")
        hi = torch.spmm(adj, input)
        if self.variant:
            # Variant case: concatenate hi and h0 along dimension 1
            support = torch.cat([hi, h0], dim=1)
            r = (1-alpha)*hi + alpha*h0
        else:
            # Default case: combine hi and h0 with alpha
            support = (1-alpha)*hi + alpha*h0
            r = support
        #check the shapes of support and weight
        print(f"GCNConvII: support shape: {support.shape}, weight shape: {self.weight.shape}")
        output = theta * torch.mm(support, self.weight) + (1 - theta)*r
        if self.residual:
            # Add residual connection if enabled
            output += input
        return output
    

class GCNII(nn.Module):
    def __init__(self, in_features: int, hidden_dims: List[int], dropout: float = 0.5, lamda: float = 0.5, alpha: float = 0.1, variant: bool = False):
        super(GCNII, self).__init__()
        self.dropout = dropout
        self.lamda = lamda
        self.alpha = alpha
        self.variant = variant
        
        dims = [in_features] + hidden_dims
        self.gcn_layers = nn.ModuleList()
        
        for i in range(hidden_dims):
            self.gcn_layers.append(GCNConvII(hidden_dims[i], hidden_dims[i + 1]))

        self.fc_in = nn.Linear(in_features, hidden_dims[0])
        
        #self.fc_in = nn.Linear(in_features, hidden_dims[0])
        #self.fc_out = nn.Linear(hidden_dims[-2], hidden_dims[-1])
        #self.act_fn = nn.ReLU()

    def forward(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Initial linear transformation
        print(f"GCNII: x shape: {x.shape}")
        h0 = self.fc_in(x)
        x = h0
        print(f"GCNII: h0 shape: {h0.shape}")
        # iterate over the GCN layers
        for i, layer in enumerate(self.gcn_layers[:-1]):
            adj = adjacency[-(i + 1)] if isinstance(adjacency, list) else adjacency
            x = torch.relu(layer(x, adj, h0, self.lamda, self.alpha, i + 1))
            x = F.dropout(x, self.dropout, training=self.training)

        adj = adjacency[0] if isinstance(adjacency, list) else adjacency
        logits = self.gcn_layers[-1](x, adj, h0, self.lamda, self.alpha, len(self.gcn_layers))
        logits = F.dropout(logits, self.dropout, training=self.training)
        
        return logits
    
    def get_intermediate_outputs(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]) -> List[torch.Tensor]:
        
        h0 = self.fc_in(x) 
        x = h0 

        intermediate_outputs = []
        target_layers = [2, 4, 8, 16, 32, 64, 128]

        for i, layer in enumerate(self.gcn_layers[:-1]):
            adj = adjacency[-(i + 1)] if isinstance(adjacency, list) else adjacency
            x = torch.relu(layer(x, adj, h0, self.lamda, self.alpha, i + 1))
            x = F.dropout(x, self.dropout)
            if i + 1 in target_layers:
                intermediate_outputs.append(x.clone())

        adj = adjacency[0] if isinstance(adjacency, list) else adjacency
        logits = self.gcn_layers[-1](x, adj, h0, self.lamda, self.alpha, len(self.gcn_layers))
        logits = F.dropout(logits, self.dropout)

        intermediate_outputs.append(logits)

        return intermediate_outputs
    
    def calculate_metrics(self, x: torch.Tensor, adjacency: Union[torch.Tensor, List[torch.Tensor]]):
        adj = adjacency if not isinstance(adjacency, list) else adjacency.pop(0)
        energy1 = calculate_dirichlet(x, adj)
        energy2 = calculate_dirichlet_energy(x, adj)
        return energy1, energy2


class GAT(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_dims: list[int]):
        super(GAT, self).__init__()

        dims = [in_features] + hidden_dims
        gat_layers = []
        for i in range(len(dims) - 1):
            gat_layers.append(GATConv(in_channels=dims[i],
                                      out_channels=dims[i + 1]))
            
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

    def intermediate_outputs(self,
                             x: torch.Tensor,
                             edge_index: Union[torch.Tensor, list[torch.Tensor]],
                             ) -> List[torch.Tensor]:
        
        intermediate_outputs = []
        target_layers = [2, 4, 8, 16, 32, 64, 128]

        for i, layer in enumerate(self.gat_layers[:-1], start=1):
            edges = edge_index[-i] if type(edge_index) == list else edge_index
            x = torch.relu(layer(x, edges))
            if i in target_layers:
                intermediate_outputs.append(x.clone())

        edges = edge_index[0] if type(edge_index) == list else edge_index
        logits = self.gat_layers[-1](x, edges)

        intermediate_outputs.append(logits)

        return intermediate_outputs
    

    def calculate_metrics(self, x: torch.Tensor, adjacency: Union[torch.Tensor, list[torch.Tensor]]):
        adj = adjacency if not isinstance(adjacency, list) else adjacency.pop(0)
        energy1 = calculate_dirichlet(x, adj)
        energy2 = calculate_dirichlet_energy(x, adj)
        return energy1, energy2


        

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


    def intermediate_outputs(self,
                             x: torch.Tensor,
                             edge_index: Union[torch.Tensor, list[torch.Tensor]],
                             ) -> List[torch.Tensor]:
        
        intermediate_outputs = []
        target_layers = [2, 4, 8, 16, 32, 64, 128]

        for i, layer in enumerate(self.gatv2_layers[:-1], start=1):
            edges = edge_index[-i] if type(edge_index) == list else edge_index
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(layer(x, edges))
            if i in target_layers:
                intermediate_outputs.append(x.clone())

        edges = edge_index[0] if type(edge_index) == list else edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.gatv2_layers[-1](x, edges)

        intermediate_outputs.append(logits)

        return intermediate_outputs
    
    def calculate_metrics(self, x: torch.Tensor, adjacency: Union[torch.Tensor, list[torch.Tensor]]):
        adj = adjacency if not isinstance(adjacency, list) else adjacency.pop(0)
        energy1 = calculate_dirichlet(x, adj)
        energy2 = calculate_dirichlet_energy(x, adj)
        return energy1, energy2
    

    
    
    


    