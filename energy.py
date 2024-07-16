    
    dirichlet_energies = {2: [], 4: [], 8: [], 16: [], 32: [], 64: [], -1: []}
    mads = {2: [], 4: [], 8: [], 16: [], 32: [], 64: [], -1: []}
    
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
        # check intermediate outputs shape

        for layer_num, intermediate_output in zip([2, 4, 8, 16, 32, 64, -1], intermediate_outputs):
            print(intermediate_output.shape)
            energy1, energy2, mad = gcn_c.calculate_metrics(intermediate_output, adj_matrices)
            dirichlet_energies[layer_num].append((energy1, energy2))
            mads[layer_num].append(mad)
            print(f'Layer {layer_num}: Energy 1: {energy1:.6f}, Energy 2: {energy2:.6f}, MAD: {mad:.6f}')

    for layer_num in [2, 4, 8, 16, 32, 64, -1]:
        avg_energy1 = sum(e[0] for e in dirichlet_energies[layer_num]) / len(dirichlet_energies[layer_num])
        avg_energy2 = sum(e[1] for e in dirichlet_energies[layer_num]) / len(dirichlet_energies[layer_num])
        avg_mad = sum(mads[layer_num]) / len(mads[layer_num])

        wandb.log({f'avg_dirichlet_energy_1_{layer_num}': avg_energy1,
                   f'avg_dirichlet_energy_2_{layer_num}': avg_energy2,
                   f'avg_mad_{layer_num}': avg_mad})
        
        logger.info(f'Final Dirichlet Energy 1 at layer {layer_num}: {avg_energy1:.6f}, '
                    f'Final Dirichlet Energy 2 at layer {layer_num}: {avg_energy2:.6f}, '
                    f'Final MAD at layer {layer_num}: {avg_mad:.6f}')
    