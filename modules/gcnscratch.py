import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = torch.sparse.mm(adj, x)
        x = torch.mm(x, self.weight) + self.bias
        x = self.batch_norm(x)
        return x

class GCN(nn.Module):
    def __init__(self, in_features, hidden_dims, dropout=0.):
        super(GCN, self).__init__()
        dims = [in_features] + hidden_dims
        self.gcn_layers = nn.ModuleList([GCNLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = dropout

    def forward(self, x, adj):
        # Add self-loops to the adjacency matrix
        adj = adj.to_dense() + torch.eye(adj.size(0))

        for layer in self.gcn_layers:
            x = F.leaky_relu(layer(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class ResGCN(nn.Module):
    def __init__(self, in_features, hidden_dims, dropout=0.):
        super(ResGCN, self).__init__()
        dims = [in_features] + hidden_dims
        self.gcn_layers = nn.ModuleList([GCNLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = dropout
        self.residual_transforms = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) if dims[i] != dims[i + 1] else None for i in range(len(dims) - 1)])

    def forward(self, x, adj):
        # Add self-loops to the adjacency matrix
        adj = adj.to_dense() + torch.eye(adj.size(0))

        for layer, residual_transform in zip(self.gcn_layers, self.residual_transforms):
            x_res = x
            x = F.leaky_relu(layer(x, adj))
            x = F.dropout(x, p=self.dropout, training=self.training)
            if residual_transform is not None:
                x_res = residual_transform(x_res)
            x = x + x_res
        return x
    

# convert edge_index to adjacency matrix
def edge_index_to_adj(edge_index, num_nodes):
    edge_index = edge_index.long()
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes))
    return adj
    
def test_gcn():
    in_features = 10
    hidden_dims = [16, 8]
    dropout = 0.5
    num_nodes = 100
    num_edges = 200

    # Randomly generate node features and edge index
    x = torch.randn(num_nodes, in_features)
    edge_index = torch.randint(num_nodes, (2, num_edges))

    # Convert edge index to adjacency matrix
    adj = edge_index_to_adj(edge_index, num_nodes)

    # Create a GCN model and a ResGCN model
    gcn = GCN(in_features, hidden_dims, dropout)
    res_gcn = ResGCN(in_features, hidden_dims, dropout)

    # Pass the node features and adjacency matrix through the models
    out_gcn = gcn(x, adj)
    out_res_gcn = res_gcn(x, adj)

    print("GCN output shape:", out_gcn.shape)
    print("ResGCN output shape:", out_res_gcn.shape)



if __name__ == '__main__':
    test_gcn()
# Test the implementation
# test_gcn()
# Expected output:

    