import argparse
from cgi import test
from math import log
import os


import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch_geometric
import wandb
from sklearn.metrics import accuracy_score, f1_score
from tap import Tap
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from modules.data import get_data, get_ppi
from modules.gcn123 import GCN, ResGCN, GCNII
from modules.utils import (TensorMap, get_logger, get_neighborhoods,
                           sample_neighborhoods_from_probs, slice_adjacency, convert_edge_index_to_adj_sparse)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Arguments(Tap):
    dataset: str = 'cora'
    sampling_hops: int = 2
    num_samples: int = 16
    lr_gc: float = 1e-3
    use_indicators: bool = True
    lr_gf: float = 1e-4
    loss_coef: float = 1e4
    log_z_init: float = 0.
    reg_param: float = 0.
    dropout: float = 0.
    model_type: str = 'gcn'
    hidden_dim: int = 256
    max_epochs: int = 30
    batch_size: int = 512
    eval_frequency: int = 1
    num_eval_batches: int = 10
    eval_on_cpu: bool = False
    eval_full_batch: bool = False
    runs: int = 10
    notes: str = None
    log_wandb: bool = True
    config_file: str = None

def sample_subgraph(data, batch_nodes, edge_index, num_hops):
    all_nodes = batch_nodes.clone().tolist()
    all_edges = []

    adjacency = sp.csr_matrix((np.ones(data.num_edges, dtype=bool),
                               data.edge_index),
                              shape=(data.num_nodes, data.num_nodes)) # NOT the normalized adjacency

    for _ in range(num_hops):
        all_nodes_tensor = torch.tensor(all_nodes)
        neighbors = get_neighborhoods(all_nodes_tensor, adjacency)
        batch_nodes = torch.unique(neighbors.view(-1))
        all_nodes += batch_nodes.tolist()
        edges = torch.stack([neighbors[0], neighbors[1]], dim=0)
        all_edges.append(edges)

    all_nodes = torch.unique(torch.tensor(all_nodes))
    all_edges = torch.cat(all_edges, dim=1)
    
     # Create a mapping from node IDs to local indices
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes.tolist())}

    # Remap edge indices
    all_edges_remapped = torch.stack([torch.tensor([node_to_idx[node.item()] for node in edge])for edge in all_edges], dim=0)


    sub_adj = convert_edge_index_to_adj_sparse(all_edges_remapped, len(all_nodes))
    sub_x = data.x[all_nodes]
    
    return sub_adj, sub_x, all_nodes


def train(args: Arguments):
    wandb.init(project='grapes',
               entity='p-goswami',
               mode='online' if args.log_wandb else 'disabled',
               config=args.as_dict(),
               notes=args.notes)
    logger = get_logger()

    path = os.path.join(os.getcwd(), 'data', args.dataset)
    data, num_features, num_classes = get_data(root=path, name=args.dataset)

    node_map = TensorMap(size=data.num_nodes)

    if args.use_indicators:
        num_indicators = args.sampling_hops + 1
    else:
        num_indicators = 0

    if args.model_type == 'gcn':
        gcn_c = GCN(data.num_features, hidden_dims=[args.hidden_dim] * 8 + [num_classes], dropout=args.dropout).to(device)

    optimizer_c = Adam(gcn_c.parameters(), lr=args.lr_gc)

    if data.y.dim() == 1:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    train_idx = data.train_mask.nonzero().squeeze(1)
    train_loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size, shuffle=True)

    val_idx = data.val_mask.nonzero().squeeze(1)
    val_loader = DataLoader(TensorDataset(val_idx), batch_size=args.batch_size)

    test_idx = data.test_mask.nonzero().squeeze(1)
    test_loader = DataLoader(TensorDataset(test_idx), batch_size=args.batch_size)

    adj = convert_edge_index_to_adj_sparse(data.edge_index, data.num_nodes)
    adjacency = sp.csr_matrix((np.ones(data.num_edges, dtype=bool),
                               data.edge_index),
                              shape=(data.num_nodes, data.num_nodes))

    logger.info('Training')
    for epoch in range(1, args.max_epochs + 1):
        gcn_c.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as bar:
            for batch in train_loader:
                batch_nodes = batch[0]
                sub_adj, sub_x, all_nodes = sample_subgraph(data, batch_nodes, data.edge_index, args.sampling_hops)

                sub_adj = sub_adj.to(device)
                sub_x = sub_x.to(device)
                all_nodes = all_nodes.to(device)
                
                optimizer_c.zero_grad()
                output, _ = gcn_c(sub_x, sub_adj)

                data.y = data.y.to(device)
                
                loss = loss_fn(output, data.y[all_nodes].to(device))
                loss.backward()
                optimizer_c.step()
                
                epoch_loss += loss.item()
                bar.set_postfix(loss=epoch_loss / (batch[0].size(0)))
                bar.update(1)

        bar.close()

        if (epoch + 1) % args.eval_frequency == 0:

            # full batch evaluation
            data.x = data.x.to(device)
            adj = adj.to(device)
            logits_total, _ = gcn_c(data.x, adj)
            val_predictions = torch.argmax(logits_total, dim=1)[data.val_mask].cpu().numpy()
            targets = data.y[data.val_mask].cpu().numpy()
            f1 = f1_score(targets, val_predictions, average='micro')

            log_dict = {'epoch': epoch,
                        'valid_f1': f1}
            
            wandb.log(log_dict)

    # test
    adj = adj.to(device)
    logits_total, _ = gcn_c(data.x, adj)
    test_predictions = torch.argmax(logits_total, dim=1)[data.test_mask].cpu()
    targets = data.y[data.test_mask]
    test_f1 = f1_score(targets, test_predictions, average='micro')

    wandb.log({'test_f1': test_f1})
    logger.info(f'Test F1: {test_f1:.3f}')



    # Compute Dirichlet energies for specified layers at the end of training
    layer_nums = [2, 4, 8, -1]
    dirichlet_energies = {layer_num: [] for layer_num in layer_nums}
    #mads = {layer_num: [] for layer_num in layer_nums}

    # move the model to CPU
    gcn_c = gcn_c.cpu()
    x = data.x.cpu()
    adj = adj.cpu()

    intermediate_outputs = gcn_c.get_intermediate_outputs(x, adj)

    # get intermediate for val set
    # check shape of intermediate_outputs

    #intermediate_outputs = [intermediate_output[val_idx].cpu() for intermediate_output in intermediate_outputs]

    # calculate metrics for specified layers for
    for layer_num, intermediate_output in zip(layer_nums, intermediate_outputs):
        energy1, energy2 = gcn_c.calculate_metrics(intermediate_output, adj)
        dirichlet_energies[layer_num].append((energy1, energy2))
        #mads[layer_num].append(mad)

    for layer_num in layer_nums:
        avg_energy1 = sum(e[0] for e in dirichlet_energies[layer_num]) / len(dirichlet_energies[layer_num])
        avg_energy2 = sum(e[1] for e in dirichlet_energies[layer_num]) / len(dirichlet_energies[layer_num])
        #avg_mad = sum(mads[layer_num]) / len(mads[layer_num])
        wandb.log({f'avg_dirichlet_energy_1_{layer_num}': avg_energy1,
                   f'avg_dirichlet_energy_2_{layer_num}': avg_energy2})
        logger.info(f'Final Dirichlet Energy 1 at layer {layer_num}: {avg_energy1:.6f}, '
                    f'Final Dirichlet Energy 2 at layer {layer_num}: {avg_energy2:.6f}')



    return test_f1




args = Arguments(explicit_bool=True).parse_args()

# If a config file is specified, load it, and parse again the CLI
# which takes precedence
if args.config_file is not None:
    args = Arguments(explicit_bool=True, config_files=[args.config_file])
    args = args.parse_args()

results = torch.empty(args.runs, 3)
for r in range(args.runs):
    test_f1 = train(args)
    results[r, 0] = test_f1

print(f'Acc: {100 * results[:,0].mean():.2f} ± {100 * results[:,0].std():.2f}')
