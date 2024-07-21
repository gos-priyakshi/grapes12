import argparse
import re

import torch
import torch_geometric
from modules.utils import (convert_edge_index_to_adj_sparse, normalize_laplacian)

from modules.utils import TensorMap, get_logger, get_neighborhoods, slice_adjacency, sample_neighborhoods_from_probs

def energy_sampling(args, gcn_gf, gcn_c, val_loader, node_map, mask, data, adjacency, num_indicators, layer_nums, device):

    #layer_nums = [2, 4, 8, -1]
    dirichlet_energies = {layer_num: [] for layer_num in layer_nums}

    
    prev_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    batch_nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    indicator_features = torch.zeros((data.num_nodes, num_indicators))
    
    for batch_idx, batch in enumerate(val_loader):

        #if batch_idx >= 5:
        #    break

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
        # check intermediate outputs shape

        for layer_num, intermediate_output in zip(layer_nums, intermediate_outputs):
            #print(intermediate_output.shape)
            energy1, energy2 = gcn_c.calculate_metrics(intermediate_output, adj_matrices)
            dirichlet_energies[layer_num].append((energy1, energy2))
            #print(f'Layer {layer_num}: Energy 1: {energy1:.6f}, Energy 2: {energy2:.6f}')


    #for layer_num in layer_nums:
    #    avg_energy1 = sum(e[0] for e in dirichlet_energies[layer_num]) / len(dirichlet_energies[layer_num])
    #    avg_energy2 = sum(e[1] for e in dirichlet_energies[layer_num]) / len(dirichlet_energies[layer_num])

    #    print(f'Layer {layer_num}: Avg Energy 1: {avg_energy1:.6f}, Avg Energy 2: {avg_energy2:.6f}')

    return dirichlet_energies    


def energy_full_batch(args, gcn_c, data, layer_nums):

    #layer_nums = [2, 4, 8, -1]
    dirichlet_energies = {layer_num: [] for layer_num in layer_nums}
    #mads = {layer_num: [] for layer_num in layer_nums}

    # full batch message passing for evaluation
    edge_index = data.edge_index
    x = data.x

    # eval on cpu
    x = x.cpu()
    edge_index = edge_index.cpu()
    gcn_c = gcn_c.cpu()

    if isinstance(edge_index, list):
        edge_indices = edge_index
    else:
        edge_indices = [edge_index for _ in range(args.sampling_hops)]

    # convert edge indices to adjacency matrices
    num_nodes = data.num_nodes
    adj_mat = [convert_edge_index_to_adj_sparse(e, num_nodes) for e in edge_indices]

    # get intermediate outputs
    intermediate_outputs = gcn_c.get_intermediate_outputs(x, adj_mat)

    # calculate metrics for specified layers
    for layer_num, intermediate_output in zip(layer_nums, intermediate_outputs):
        energy1, energy2 = gcn_c.calculate_metrics(intermediate_output, adj_mat)
        dirichlet_energies[layer_num].append((energy1, energy2))
        #mads[layer_num].append(mad)


    #for layer_num in layer_nums:
    #    avg_energy1 = sum(e[0] for e in dirichlet_energies[layer_num]) / len(dirichlet_energies[layer_num])
    #    avg_energy2 = sum(e[1] for e in dirichlet_energies[layer_num]) / len(dirichlet_energies[layer_num])
        #avg_mad = sum(mads[layer_num]) / len(mads[layer_num])

    return dirichlet_energies

    