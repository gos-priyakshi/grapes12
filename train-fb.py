# Full batch without pytorch geometric

import argparse
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, f1_score
from tap import Tap
from torch.distributions import Bernoulli
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from modules.data import get_data, get_ppi
from modules.gcn123 import GCN, ResGCN
from modules.utils import (TensorMap, get_logger, get_neighborhoods,
                           sample_neighborhoods_from_probs, slice_adjacency)


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

    node_map = TensorMap(size=data.num_nodes)

    if args.use_indicators:
        num_indicators = args.sampling_hops + 1
    else:
        num_indicators = 0

    if args.model_type == 'gcn':
        model = GCN(data.num_features, hidden_dims=[args.hidden_dim, num_classes], dropout=args.dropout).to(device)
    elif args.model_type == 'resgcn':
        model = ResGCN(data.num_features, hidden_dims=[args.hidden_dim, num_classes], dropout=args.dropout).to(device)
    else:
        raise ValueError(f'Invalid model type: {args.model_type}')

    optimizer = Adam(model.parameters(), lr=args.lr_gc)

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

    adjacency = sp.csr_matrix((np.ones(data.num_edges, dtype=bool),
                               data.edge_index),
                              shape=(data.num_nodes, data.num_nodes))
    # Convert the adjacency matrix to a dense numpy array
    adjacency_dense = adjacency.toarray()

    # Convert the numpy array to a PyTorch tensor
    adjacency_tensor = torch.from_numpy(adjacency_dense).float().to(device)


    logger.info('Training')

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        x = data.x.to(device)
        out = model(x, adjacency_tensor.to(device))
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % args.eval_frequency == 0:
            model.eval()
            with torch.no_grad():
                out = model(x, adjacency_tensor.to(device))
                pred = out.argmax(dim=1)
                #train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
                #val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()

                train_pred = pred[data.train_mask].cpu().numpy()
                train_targets = data.y[data.train_mask].cpu().numpy()
                train_acc = accuracy_score(train_targets, train_pred)

                val_pred = pred[data.val_mask].cpu().numpy()
                val_targets = data.y[data.val_mask].cpu().numpy()
                val_acc = accuracy_score(val_targets, val_pred)

                # Calculate validation F1 score
                val_predictions = pred[data.val_mask].cpu()
                targets = data.y[data.val_mask]
                val_f1 = f1_score(targets, val_predictions, average='micro')

                log_dict = {'epoch': epoch,
                            'loss': loss.item(),
                            'train_accuracy': train_acc,
                            'valid_accuracy': val_acc,
                            'valid_f1': val_f1}
                


                wandb.log(log_dict)



    x = data.x.to(device)
    logits = model(x, adjacency_tensor.to(device))
    test_predictions = torch.argmax(logits, dim=1)[data.test_mask].cpu()
    targets = data.y[data.test_mask]
    test_accuracy = accuracy_score(targets, test_predictions)
    test_f1 = f1_score(targets, test_predictions, average='micro')

    wandb.log({'test_accuracy': test_accuracy,
               'test_f1': test_f1})
    logger.info(f'test_accuracy={test_accuracy:.3f}, '
                f'test_f1={test_f1:.3f}')

    return test_f1

# minibatch training

def train_minibatch(args: Arguments):
    wandb.init(project='grapes',
               entity='p-goswami',
               mode='online' if args.log_wandb else 'disabled',
               config=args.as_dict(),
               notes=args.notes)
    logger = get_logger()

    path = os.path.join(os.getcwd(), 'data', args.dataset)
    data, num_features, num_classes = get_data(root=path, name=args.dataset)

    train_idx = data.train_mask.nonzero().squeeze(1)
    train_loader = DataLoader(TensorDataset(train_idx), batch_size=args.batch_size)

    val_idx = data.val_mask.nonzero().squeeze(1)
    val_loader = DataLoader(TensorDataset(val_idx), batch_size=args.batch_size)

    test_idx = data.test_mask.nonzero().squeeze(1)
    test_loader = DataLoader(TensorDataset(test_idx), batch_size=args.batch_size)

    node_map = TensorMap(size=data.num_nodes)

    if args.use_indicators:
        num_indicators = args.sampling_hops + 1
    else:
        num_indicators = 0

    if args.model_type == 'gcn':
        model = GCN(data.num_features, hidden_dims=[args.hidden_dim, num_classes], dropout=args.dropout).to(device)
    elif args.model_type == 'resgcn':
        model = ResGCN(data.num_features, hidden_dims=[args.hidden_dim, num_classes], dropout=args.dropout).to(device)
    else:
        raise ValueError(f'Invalid model type: {args.model_type}')

    optimizer = Adam(model.parameters(), lr=args.lr_gc)

    if data.y.dim() == 1:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    adjacency = sp.csr_matrix((np.ones(data.num_edges, dtype=bool),
                               data.edge_index),
                              shape=(data.num_nodes, data.num_nodes))

    logger.info('Training')

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            batch_nodes = batch[0]
            x = data.x[batch_nodes].to(device)

            # Convert adjacency matrix to PyTorch tensor
            adjacency_tensor = torch.from_numpy(adjacency.toarray()).float().to(device)

            out = model(x, adjacency_tensor)
            loss = loss_fn(out, data.y[batch_nodes].to(device))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        if epoch % args.eval_frequency == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for val_batch in val_loader:
                    val_batch_nodes = val_batch[0]
                    val_x = data.x[val_batch_nodes].to(device)
                    val_out = model(val_x, adjacency_tensor)
                    val_loss += loss_fn(val_out, data.y[val_batch_nodes].to(device)).item()
                val_loss /= len(val_loader)

                logger.info(f'Epoch [{epoch}/{args.max_epochs}], '
                            f'Training Loss: {epoch_loss:.4f}, '
                            f'Validation Loss: {val_loss:.4f}')

                wandb.log({'epoch': epoch, 'train_loss': epoch_loss, 'val_loss': val_loss})

    # Test
    test_loss = 0
    with torch.no_grad():
        for test_batch in test_loader:
            test_batch_nodes = test_batch[0]
            test_x = data.x[test_batch_nodes].to(device)
            test_out = model(test_x, adjacency_tensor)
            test_loss += loss_fn(test_out, data.y[test_batch_nodes].to(device)).item()
        test_loss /= len(test_loader)

        logger.info(f'Test Loss: {test_loss:.4f}')
        wandb.log({'test_loss': test_loss})

    return test_loss


#args = Arguments(explicit_bool=True).parse_args()



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