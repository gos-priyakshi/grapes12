from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# GCN implementation wihout using torch geometric


class Graph_Convolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(Graph_Convolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self): # Initialize the weights and biases
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(x, self.weight) # Transform the input features
        row, col = edge_index # row and col are the source and target nodes
        out = torch.zeros(x.size(0), self.out_features, device=x.device) # Initialize the output tensor
        out[row] += x[col] # Aggregate the features of the neighbors
        out = out + self.bias # Add the bias term
        return F.relu(out) # Apply the activation function
    

class GCN(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_dims: list[int], dropout: float=0.):
        super(GCN, self).__init__()
        self.dropout = dropout
        dims = [in_features] + hidden_dims
        gcn_layers = []
        for i in range(len(hidden_dims) - 1):
            gcn_layers.append(Graph_Convolution(in_features=dims[i],
                                      out_features=dims[i + 1]))

        gcn_layers.append(Graph_Convolution(in_features=dims[-2], out_features=dims[-1]))
        self.gcn_layers = nn.ModuleList(gcn_layers)

    def forward(self,
                x: torch.Tensor,
                edge_index: Union[torch.Tensor, list[torch.Tensor]],
                ) -> torch.Tensor:
        layerwise_adjacency = type(edge_index) == list

        for i, layer in enumerate(self.gcn_layers[:-1], start=1):
            if abs(i) > len(edge_index):
                print(f"Error: Trying to access index {-i} but edge_index only has {len(edge_index)} elements.")
                # Handle the error here, for example by skipping this iteration with 'continue'
                continue
            edges = edge_index[-i] if layerwise_adjacency else edge_index
            x = layer(x, edges)
            x = F.dropout(x, p=self.dropout, training=self.training)

        edges = edge_index[0] if layerwise_adjacency else edge_index
        logits = self.gcn_layers[-1](x, edges)
        logits = F.dropout(logits, p=self.dropout, training=self.training)

        # torch.cuda.synchronize()
        memory_alloc = torch.cuda.memory_allocated() / (1024 * 1024)

        return logits, memory_alloc
    


# GAT implementation wihout using torch geometric

class Graph_Attention(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(Graph_Attention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.att = nn.Parameter(torch.Tensor(1, 2 * out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(x, self.weight)
        row, col = edge_index
        out = torch.zeros(x.size(0), self.out_features, device=x.device)
        out[row] += x[col]
        out = out + self.bias
        out = F.relu(out)
        alpha = (torch.cat([out[row], out[col]], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = F.softmax(alpha, dim=0)
        out = out * alpha.unsqueeze(-1)
        return out
    

class GAT(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_dims: list[int], dropout: float=0.):
        super(GAT, self).__init__()
        self.dropout = dropout
        dims = [in_features] + hidden_dims
        gat_layers = []
        for i in range(len(hidden_dims) - 1):
            gat_layers.append(Graph_Attention(in_features=dims[i],
                                      out_features=dims[i + 1]))

        gat_layers.append(Graph_Attention(in_features=dims[-2], out_features=dims[-1]))
        self.gat_layers = nn.ModuleList(gat_layers)

    def forward(self,
                x: torch.Tensor,
                edge_index: Union[torch.Tensor, list[torch.Tensor]],
                ) -> torch.Tensor:
        layerwise_adjacency = type(edge_index) == list

        for i, layer in enumerate(self.gat_layers[:-1], start=1):
            edges = edge_index[-i] if layerwise_adjacency else edge_index
            x = layer(x, edges)

        edges = edge_index[0] if layerwise_adjacency else edge_index
        logits = self.gat_layers[-1](x, edges)

        # memory
        memory_alloc = torch.cuda.memory_allocated() / (1024 * 1024)

        return logits, memory_alloc
    

# ResGCN implementation wihout using torch geometric

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
            gcn_layers.append(Graph_Convolution(in_features=dims[i], out_features=dims[i + 1]))
        self.gcn_layers = nn.ModuleList(gcn_layers)

    def forward(self,
                x: torch.Tensor,
                edge_index: Union[torch.Tensor, list[torch.Tensor]],
                ) -> torch.Tensor:
        layerwise_adjacency = type(edge_index) == list
        x = self.transform(x)
        for i, layer in enumerate(self.gcn_layers):
            edges = edge_index[-i] if layerwise_adjacency else edge_index
            x = x + layer(x, edges)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # memory
        memory_alloc = torch.cuda.memory_allocated() / (1024 * 1024)

        return x, memory_alloc
    

