import argparse
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch_geometric
import wandb
from sklearn.metrics import accuracy_score, f1_score
from tap import Tap
from torch.distributions import Bernoulli
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

    sampling_hops: int = 129
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

    runs: int = 1
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

    node_map = TensorMap(size=data.num_nodes)

    if args.use_indicators:
        num_indicators = args.sampling_hops + 1
    else:
        num_indicators = 0

    if args.model_type == 'gcn':
        gcn_c = ResGCN(data.num_features, hidden_dims=[args.hidden_dim] * 32 + [num_classes], dropout=args.dropout).to(device)

    optimizer_c = Adam(gcn_c.parameters(), lr=args.lr_gc)

    if data.y.dim() == 1:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    train_idx = data.train_mask.nonzero().squeeze(1)
    train_loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size)

    val_idx = data.val_mask.nonzero().squeeze(1)
    val_loader = DataLoader(TensorDataset(val_idx), batch_size=args.batch_size)

    test_idx = data.test_mask.nonzero().squeeze(1)
    test_loader = DataLoader(TensorDataset(test_idx), batch_size=args.batch_size)

    #adjacency = sp.csr_matrix((np.ones(data.num_edges, dtype=bool),
    #                           data.edge_index),
    #                          shape=(data.num_nodes, data.num_nodes))
    
    # convert edge index to adjacency matrix
    adj = convert_edge_index_to_adj_sparse(data.edge_index, data.num_nodes)

    logger.info('Training')
    for epoch in range(1, args.max_epochs + 1):
        acc_loss_gfn = 0
        acc_loss_c = 0
        acc_loss_binom = 0

        with tqdm(total=1, desc=f'Epoch {epoch}') as bar:
            x = data.x.to(device)
            logits, gcn_mem_alloc = gcn_c(x, adj.to(device))
            loss_c = loss_fn(logits[data.train_mask], data.y[data.train_mask].to(device))

            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()

            wandb.log({'loss_c': loss_c.item()})

            bar.set_postfix({'loss_c': loss_c.item()})
            bar.update()

        bar.close()

        if (epoch + 1) % args.eval_frequency == 0:
            val_predictions = torch.argmax(logits, dim=1)[data.val_mask].cpu()
            targets = data.y[data.val_mask]
            accuracy = accuracy_score(targets, val_predictions)
            f1 = f1_score(targets, val_predictions, average='micro')

            log_dict = {'epoch': epoch,
                        'valid_f1': f1}

            logger.info(f'loss_c={acc_loss_c:.6f}, '
                        f'valid_f1={f1:.3f}')
            wandb.log(log_dict)

    #x = data.x.to(device)
    logits, gcn_mem_alloc = gcn_c(x, adj.to(device))
    test_predictions = torch.argmax(logits, dim=1)[data.test_mask].cpu()
    targets = data.y[data.test_mask]
    test_accuracy = accuracy_score(targets, test_predictions)
    test_f1 = f1_score(targets, test_predictions, average='micro')

    
    wandb.log({'test_accuracy': test_accuracy,
               'test_f1': test_f1})
    logger.info(f'test_accuracy={test_accuracy:.3f}, '
                f'test_f1={test_f1:.3f}')

    
    # Compute Dirichlet energies and MAD for specified layers at the end of training
    layer_nums = [2, 4, 8, 16, 32, -1]
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

    
    #energy1, energy2, mad = gcn_c.calculate_metrics(logits, adj) 


    return test_f1


args = Arguments(explicit_bool=True).parse_args()

# If a config file is specified, load it, and parse again the CLI
# which takes precedence
if args.config_file is not None:
    args = Arguments(explicit_bool=True, config_files=[args.config_file])
    args = args.parse_args()

results = torch.empty(args.runs, 3)
for r in range(args.runs):
    test_f1= train(args)
    results[r, 0] = test_f1

print(f'Acc: {100 * results[:,0].mean():.2f} ± {100 * results[:,0].std():.2f}')
