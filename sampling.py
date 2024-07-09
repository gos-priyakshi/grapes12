import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import wandb
from tap import Tap
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
import random

from evalnew import evaluate
from modules.data import get_data
from modules.gcn123 import GCN, ResGCN
from modules.utils import (TensorMap, get_logger, get_neighborhoods,
                           sample_neighborhoods_from_probs, slice_adjacency, convert_edge_index_to_adj_sparse, normalize_laplacian)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")


class Arguments(Tap):
    dataset: str = 'cora'

    sampling_hops: int = 2
    num_samples: int = 64
    use_indicators: bool = True
    lr_gf: float = 1e-4 # learning rate for the GCN-GF model
    lr_gc: float = 1e-3 # learning rate for the GCN model
    loss_coef: float = 1e4 # coefficient for the loss of the GCN model
    log_z_init: float = 0. # initial value for the log partition function
    reg_param: float = 0. # regularization parameter for the GCN model
    dropout: float = 0.

    model_type: str = 'gcn' 
    hidden_dim: int = 256
    max_epochs: int = 30
    batch_size: int = 512
    eval_frequency: int = 1
    num_eval_batches: int = 10
    eval_on_cpu: bool = True 
    eval_full_batch: bool = True
   
    runs: int = 10
    notes: str = None
    log_wandb: bool = True
    config_file: str = None


def train(args: Arguments):
    wandb.init(project='grapes', # change these to my own wandb project
               entity='p-goswami',
               mode='online' if args.log_wandb else 'disabled',
               config=args.as_dict(),
               notes=args.notes)
    logger = get_logger() # what does this do? 

    path = os.path.join(os.getcwd(), 'data', args.dataset) 
    data, num_features, num_classes = get_data(root=path, name=args.dataset)

    yelp_single=False
    if yelp_single:
        label_frequency = data.y.sum(dim=0)
        y = torch.empty(data.num_nodes, dtype=torch.long)
        for i in range(0, data.num_nodes):
            labels = data.y[i].nonzero(as_tuple=False)
            if labels.numel() == 0:
                y[i] = torch.tensor([2])
            else:
                y[i] = labels[torch.abs(label_frequency[labels].squeeze() - data.num_nodes / 2).argmin()]

        lbl_uni, lbl_cnt = torch.unique(y, return_counts=True)
        lbl_map = TensorMap(size=lbl_uni.max()+1)
        lbl_map.update(lbl_uni)
        lbl_new = lbl_map.map(y)
        data.y = lbl_new
        num_classes = len(lbl_uni)

    node_map = TensorMap(size=data.num_nodes) 

    if args.use_indicators:
        num_indicators = args.sampling_hops + 1
    else:
        num_indicators = 0

    if args.model_type == 'gcn':
        gcn_c = GCN(data.num_features, hidden_dims=[args.hidden_dim] * 8 + [num_classes], dropout=args.dropout).to(device)
        # GCN model for GFlotNet sampling
        gcn_gf = GCN(data.num_features + num_indicators,
                      hidden_dims=[args.hidden_dim, 1]).to(device)

    log_z = torch.tensor(args.log_z_init, requires_grad=True)
    optimizer_c = Adam(gcn_c.parameters(), lr=args.lr_gc)
    optimizer_gf = Adam(list(gcn_gf.parameters()) + [log_z], lr=args.lr_gf)

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
                              shape=(data.num_nodes, data.num_nodes)) # NOT the normalized adjacency

    prev_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    batch_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    indicator_features = torch.zeros((data.num_nodes, num_indicators))

    # This will collect memory allocations for all epochs
    all_mem_allocations_point1 = []
    all_mem_allocations_point2 = []
    all_mem_allocations_point3 = []

    logger.info('Training')
    for epoch in range(1, args.max_epochs + 1):
        acc_loss_gfn = 0
        acc_loss_c = 0
        # add a list to collect memory usage
        mem_allocations_point1 = []  # The first point of memory usage measurement after the GCNConv forward pass
        mem_allocations_point2 = []  # The second point of memory usage measurement after the GCNConv backward pass
        mem_allocations_point3 = []  # The third point of memory usage measurement after the GCNConv backward pass

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as bar:
            for batch_id, batch in enumerate(train_loader):
                # torch.cuda.empty_cache()
                # torch.cuda.reset_peak_memory_stats()

                target_nodes = batch[0]

                previous_nodes = target_nodes.clone()
                all_nodes_mask = torch.zeros_like(prev_nodes_mask)
                all_nodes_mask[target_nodes] = True
            
                indicator_features.zero_()
                indicator_features[target_nodes, -1] = 1.0

                global_edge_indices = []
                log_probs = []
                sampled_sizes = []
                neighborhood_sizes = []
                all_statistics = []
                # Sample neighborhoods with the GCN-GF model
                for hop in range(args.sampling_hops):
                    # Get neighborhoods of target nodes in batch
                    neighborhoods = get_neighborhoods(previous_nodes, adjacency)

                    # Identify batch nodes (nodes + neighbors) and neighbors
                    prev_nodes_mask.zero_()
                    batch_nodes_mask.zero_()
                    prev_nodes_mask[previous_nodes] = True
                    batch_nodes_mask[neighborhoods.view(-1)] = True
                    neighbor_nodes_mask = batch_nodes_mask & ~prev_nodes_mask

                    #check if the device is correct
                    #print(batch_nodes_mask.device)
                    #print(neighbor_nodes_mask.device)
                    #print(indicator_features.device)
                    #print(node_map.values.device)

                    
                    batch_nodes = node_map.values[batch_nodes_mask]
                    neighbor_nodes = node_map.values[neighbor_nodes_mask]
                    indicator_features[neighbor_nodes, hop] = 1.0

                    # Map neighborhoods to local node IDs
                    node_map.update(batch_nodes)
                    local_neighborhoods = node_map.map(neighborhoods).to(device)


                    # and size
                    #print(local_neighborhoods.size())
                    # number of unique nodes in the local neighborhood
                    #print(len(torch.unique(local_neighborhoods)))

                    # convert to adjacency matrix
                    num_nodes = len(torch.unique(local_neighborhoods))
                    
                    
                    #start = time.time()
                    local_neighborhoods = convert_edge_index_to_adj_sparse(local_neighborhoods, num_nodes) ## torch coo tensor
                    #print('time', time.time() - start)   

                    if args.use_indicators:
                        x = torch.cat([data.x[batch_nodes],
                                       indicator_features[batch_nodes]],
                                      dim=1
                                      ).to(device)
                    else:
                        x = data.x[batch_nodes].to(device)

                    # Get probabilities for sampling each node
                    node_logits, _ = gcn_gf(x, local_neighborhoods) 
                    # Select logits for neighbor nodes only
                    node_logits = node_logits[node_map.map(neighbor_nodes)]

                    # Sample neighbors using the logits
                    sampled_neighboring_nodes, log_prob, statistics = sample_neighborhoods_from_probs(
                        node_logits,
                        neighbor_nodes,
                        args.num_samples
                    )
                    all_nodes_mask[sampled_neighboring_nodes] = True

                    log_probs.append(log_prob)
                    sampled_sizes.append(sampled_neighboring_nodes.shape[0])
                    neighborhood_sizes.append(neighborhoods.shape[-1])
                    all_statistics.append(statistics)

                    # Update batch nodes for next hop

                    #print(target_nodes.device)
                    #print(sampled_neighboring_nodes.device)

                    batch_nodes = torch.cat([target_nodes,
                                             sampled_neighboring_nodes],
                                            dim=0)


                    # Retrieve the edge index that results after sampling
                    k_hop_edges = slice_adjacency(adjacency,
                                                  rows=previous_nodes,
                                                  cols=batch_nodes)
                    global_edge_indices.append(k_hop_edges)

                    # Update the previous_nodes
                    previous_nodes = batch_nodes.clone()

                # Converting global indices to the local of final batch_nodes.
                # The final batch_nodes are the nodes sampled from the second
                # hop concatenated with the target nodes
                all_nodes = node_map.values[all_nodes_mask]
                node_map.update(all_nodes)
                local_edge_indices = [node_map.map(e).to(device) for e in global_edge_indices]
                # convert to adjacency matrices
                num_nodes = len(torch.unique(all_nodes))
                adj_matrices = [convert_edge_index_to_adj_sparse(e, num_nodes) for e in local_edge_indices]
                # convert adjacency to normalized laplacian
                # adj_matrices = [normalize_laplacian(e) for e in adj_matrices]

                x = data.x[all_nodes].to(device)
                logits, gcn_mem_alloc = gcn_c(x, adj_matrices)

                # calculate metrics in the last epo

                local_target_ids = node_map.map(target_nodes)
                loss_c = loss_fn(logits[local_target_ids],
                                 data.y[target_nodes].to(device)) + args.reg_param*torch.sum(torch.var(logits, dim=1))

                optimizer_c.zero_grad()

                mem_allocations_point3.append(torch.cuda.memory_allocated() / (1024 * 1024))

                loss_c.backward()

                optimizer_c.step()

                optimizer_gf.zero_grad()
                cost_gfn = loss_c.detach()

                loss_gfn = (log_z + torch.sum(torch.cat(log_probs, dim=0)) + args.loss_coef*cost_gfn)**2

                mem_allocations_point1.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
                mem_allocations_point2.append(gcn_mem_alloc)

                loss_gfn.backward()

                optimizer_gf.step()

                batch_loss_gfn = loss_gfn.item()
                batch_loss_c = loss_c.item()

                #print(next(gcn_c.parameters()).device)
                #print(next(gcn_gf.parameters()).device)

                wandb.log({'batch_loss_gfn': batch_loss_gfn,
                           'batch_loss_c': batch_loss_c,
                           'log_z': log_z,
                           '-log_probs': -torch.sum(torch.cat(log_probs, dim=0))})

                log_dict = {}
                for i, statistics in enumerate(all_statistics):
                    for key, value in statistics.items():
                        log_dict[f"{key}_{i}"] = value
                wandb.log(log_dict)

                acc_loss_gfn += batch_loss_gfn / len(train_loader)
                acc_loss_c += batch_loss_c / len(train_loader)

                bar.set_postfix({'batch_loss_gfn': batch_loss_gfn,
                                 'batch_loss_c': batch_loss_c,
                                 'log_z': log_z.item(),
                                 'log_probs': torch.sum(torch.cat(log_probs, dim=0)).item()})
                bar.update()

        bar.close()

        

        all_mem_allocations_point1.extend(mem_allocations_point1)
        all_mem_allocations_point2.extend(mem_allocations_point2)
        all_mem_allocations_point3.extend(mem_allocations_point3)

        if (epoch + 1) % args.eval_frequency == 0:
            accuracy, f1, _, _, _ = evaluate(gcn_c,
                                    gcn_gf,
                                    data,
                                    args,
                                    adjacency,
                                    node_map,
                                    num_indicators,
                                    device,
                                    data.val_mask,
                                    args.eval_on_cpu,
                                    loader=val_loader,
                                    full_batch=args.eval_full_batch,
                                    )
            if args.eval_on_cpu:
                gcn_c.to(device)

            log_dict = {'epoch': epoch,
                        'loss_gfn': acc_loss_gfn,
                        'loss_c': acc_loss_c,
                        'valid_accuracy': accuracy,
                        'valid_f1': f1}
            
            # log dirichlet energy values
            #log_dict['dirichlet_energy'] = gcn_c.get_dirichlet_energy()
            #log_dict['mad'] = gcn_c.get_mean_average_distance()

            logger.info(f'loss_gfn={acc_loss_gfn:.6f}, '
                        f'loss_c={acc_loss_c:.6f}, '
                        f'valid_accuracy={accuracy:.3f}, '
                        f'valid_f1={f1:.3f}')
            wandb.log(log_dict)

    # calculate dirichlet energy and mean average distance
    #x = data.x[all_nodes].to(device)
    #logits, _ = gcn_c(x, adj_matrices)
    #energy1, energy2, mad = gcn_c.calculate_metrics(logits, adj_matrices)

    #wandb.log({'dirichlet_energy 1': energy1,
    #           'dirichlet_energy 2': energy2,
    #           'mad': mad})

    # Calculate metrics on a few training batches after all epochs are done
    
    
    
    dirichlet_energies = {2: [], 4: [], 8: [], 16: [], 32: [], 64: [], 128: []}
    mads = {2: [], 4: [], 8: [], 16: [], 32: [], 64: [], 128: []}
    
    dirichlet_energies = []
    mads = []

    for batch_idx, batch in enumerate(train_loader):

        if batch_idx >= 5:
            break

        target_nodes = batch[0]
        previous_nodes = target_nodes.clone()
        all_nodes_mask = torch.zeros_like(prev_nodes_mask)
        all_nodes_mask[target_nodes] = True

        indicator_features.zero_()
        indicator_features[target_nodes, -1] = 1.0

        global_edge_indices = []

        for hop in range(args.sampling_hops):
            neighborhoods = get_neighborhoods(previous_nodes, adjacency)

            prev_nodes_mask.zero_()
            batch_nodes_mask.zero_()
            prev_nodes_mask[previous_nodes] = True
            batch_nodes_mask[neighborhoods.view(-1)] = True
            neighbor_nodes_mask = batch_nodes_mask & ~prev_nodes_mask

            batch_nodes = node_map.values[batch_nodes_mask]
            neighbor_nodes = node_map.values[neighbor_nodes_mask]
            indicator_features[neighbor_nodes, hop] = 1.0

            node_map.update(batch_nodes)
            local_neighborhood = node_map.map(neighborhoods).to(device)

            # convert to adjacency matrix
            num_nodes = len(torch.unique(local_neighborhood))
            local_neighborhood = convert_edge_index_to_adj_sparse(local_neighborhood, num_nodes)

            if args.use_indicators:
                x = torch.cat([data.x[batch_nodes],
                               indicator_features[batch_nodes]],
                              dim=1
                              ).to(device)
            else:
                x = data.x[batch_nodes].to(device)

            node_logits, _ = gcn_gf(x, local_neighborhood)
            node_logits = node_logits[node_map.map(neighbor_nodes)]

            sampled_neighboring_nodes, log_prob, statistics = sample_neighborhoods_from_probs(
                node_logits,
                neighbor_nodes,
                args.num_samples
            )

            all_nodes_mask[sampled_neighboring_nodes] = True
            batch_nodes = torch.cat([target_nodes, sampled_neighboring_nodes], dim=0)
            k_hop_edges = slice_adjacency(adjacency, rows=previous_nodes, cols=batch_nodes)
            global_edge_indices.append(k_hop_edges)

            previous_nodes = batch_nodes.clone()

        all_nodes = node_map.values[all_nodes_mask]
        node_map.update(all_nodes)
        local_edge_indices = [node_map.map(e).to(device) for e in global_edge_indices]
        num_nodes = len(torch.unique(all_nodes))
        adj_matrices = [convert_edge_index_to_adj_sparse(e, num_nodes) for e in local_edge_indices]

        x = data.x[all_nodes].to(device)
        intermediate_outputs = gcn_c.get_intermediate_outputs(x, adj_matrices)

        for layer_num, intermediate_output in zip([2, 4, 8, 16, 32, 64, 128], intermediate_outputs):
            energy1, energy2, mad = gcn_c.calculate_metrics(intermediate_output, adj_matrices)
            dirichlet_energies[layer_num].append((energy1, energy2))
            mads[layer_num].append(mad)

    for layer_num in [2, 4, 8, 16, 32, 64, 128]:
        avg_energy1 = sum(e[0] for e in dirichlet_energies[layer_num]) / len(dirichlet_energies[layer_num])
        avg_energy2 = sum(e[1] for e in dirichlet_energies[layer_num]) / len(dirichlet_energies[layer_num])
        avg_mad = sum(mads[layer_num]) / len(mads[layer_num])

        wandb.log({f'avg_dirichlet_energy_1_{layer_num}': avg_energy1,
                   f'avg_dirichlet_energy_2_{layer_num}': avg_energy2,
                   f'avg_mad_{layer_num}': avg_mad})
        
        logger.info(f'Final Dirichlet Energy 1 at layer {layer_num}: {avg_energy1:.6f}, '
                    f'Final Dirichlet Energy 2 at layer {layer_num}: {avg_energy2:.6f}, '
                    f'Final MAD at layer {layer_num}: {avg_mad:.6f}')
    

    test_accuracy, test_f1, e1, e2, m = evaluate(gcn_c,
                                      gcn_gf,
                                      data,
                                      args,
                                      adjacency,
                                      node_map,
                                      num_indicators,
                                      device,
                                      data.test_mask,
                                      args.eval_on_cpu,
                                      loader=test_loader,
                                      full_batch=args.eval_full_batch)

    wandb.log({'test_accuracy': test_accuracy,
               'test_f1': test_f1,
               'de1': e1,
               'de2': e2,
               'md': m})
    logger.info(f'test_accuracy={test_accuracy:.3f}, '
                f'test_f1={test_f1:.3f}')
    return test_f1, all_mem_allocations_point1, all_mem_allocations_point2, all_mem_allocations_point3


args = Arguments(explicit_bool=True).parse_args()

# If a config file is specified, load it, and parse again the CLI
# which takes precedence
if args.config_file is not None:
    args = Arguments(explicit_bool=True, config_files=[args.config_file])
    args = args.parse_args()

results = torch.empty(args.runs)
mem1 = []
mem2 = []
mem3 = []
for r in range(args.runs):
    test_f1, mean_mem1, mean_mem2, mean_mem3 = train(args)
    results[r] = test_f1
    mem1.extend(mean_mem1)
    mem2.extend(mean_mem2)
    mem3.extend(mean_mem3)


print(f'Memory point 1: {np.mean(mem1)} MB ± {np.std(mem1):.2f}')
print(f'Memory point 2: {np.mean(mem2)} MB ± {np.std(mem2):.2f}')
print(f'Memory point 2: {np.mean(mem3)} MB ± {np.std(mem3):.2f}')
print(f'Acc: {100 * results.mean():.2f} ± {100 * results.std():.2f}')
