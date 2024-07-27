import argparse
from cgi import test
from math import log
from operator import ne
import os
from platform import node
from re import sub



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


def train(args: Arguments):
    wandb.init(project='grapes',
               entity='p-goswami',
               mode='online' if args.log_wandb else 'disabled',
               config=args.as_dict(),
               notes=args.notes)
    logger = get_logger()

    path = os.path.join(os.getcwd(), 'data', args.dataset)
    data, num_features, num_classes = get_data(root=path, name=args.dataset)


    if args.use_indicators:
        num_indicators = args.sampling_hops + 1
    else:
        num_indicators = 0

    if args.model_type == 'gcn':
        gcn_c = GCN(data.num_features, hidden_dims=[args.hidden_dim] * 2 + [num_classes], dropout=args.dropout).to(device)

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
        total_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as bar:
            for batch_id, batch in enumerate(train_loader):
            

                # get the target nodes
                batch_nodes = batch[0]
                print(len(batch_nodes))
                print(batch_nodes)

                # get the adjacency matrix for the batch
                edge_ind = slice_adjacency(adjacency, rows=batch_nodes, cols=batch_nodes)
                local_adj = convert_edge_index_to_adj_sparse(edge_ind, len(batch_nodes))

                # get the features for the batch
                sub_x = data.x[batch_nodes].to(device)

                logits, _ = gcn_c(sub_x, local_adj)

                loss = loss_fn(logits, data.y[batch_nodes].to(device))
                if torch.isnan(loss):
                    print("NaN loss detected")
                    continue
                print(loss.item())

                batch_loss = loss.item()
            
                total_loss += batch_loss / len(train_loader)             

                loss.backward()
                optimizer_c.step()

                bar.set_postfix({
                    'batch_loss': batch_loss
                })
                bar.update()

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
                        'valid_f1': f1,
                        'train_loss': total_loss}
            
            wandb.log(log_dict)

    # test
    adj = adj.to(device)
    logits_total, _ = gcn_c(data.x, adj)
    test_predictions = torch.argmax(logits_total, dim=1)[data.test_mask].cpu().numpy()
    targets = data.y[data.test_mask].cpu().numpy()
    test_f1 = f1_score(targets, test_predictions, average='micro')

    wandb.log({'test_f1': test_f1})
    logger.info(f'Test F1: {test_f1:.3f}')



    # Compute Dirichlet energies for specified layers at the end of training
    layer_nums = [2, 4, 8, 16, 32, 64, -1]
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
