import torch
import numpy as np
from tqdm import tqdm


def hac_dfs_tokenization(data):
    """Hierarchical Agglomerative Clustering with DFS tokenization for graph ordering."""
    pos = data.pos
    row, col = data.edge_index

    # Compute similarity on edges
    dist = torch.norm(pos[row] - pos[col], p=2, dim=-1)
    tie_breaker = (row + col).float() * 1e-6
    dist = dist + tie_breaker
    affinity = 1.0 / (1.0 + dist)

    scores = affinity.detach().cpu().numpy()
    rows = row.detach().cpu().numpy()
    cols = col.detach().cpu().numpy()

    num_nodes = data.num_nodes

    # Initialize variables
    parent = np.arange(2 * num_nodes)
    children = {}
    active_roots = set(range(num_nodes))
    next_cluster_id = num_nodes

    def find(i):
        """Find cluster of node."""
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        """Merge clusters."""
        nonlocal next_cluster_id
        root_i = find(i)
        root_j = find(j)

        if root_i == root_j:
            return None, None, None

        new_cluster = next_cluster_id
        next_cluster_id += 1

        parent[root_i] = new_cluster
        parent[root_j] = new_cluster
        parent[new_cluster] = new_cluster

        children[new_cluster] = [root_i, root_j]

        return new_cluster, root_i, root_j

    # HAC loop
    while len(active_roots) > 1:
        best_edges = {}

        # For each edge, consider its clusters and keep cheapest edge
        for i in range(len(scores)):
            u, v, score = rows[i], cols[i], scores[i]
            root_u, root_v = find(u), find(v)

            if root_u == root_v:
                continue

            # Best edge for root_u
            if root_u not in best_edges or score > best_edges[root_u][0]:
                best_edges[root_u] = (score, root_v)

            # Best edge for root_v
            if root_v not in best_edges or score > best_edges[root_v][0]:
                best_edges[root_v] = (score, root_u)

        if not best_edges:
            break

        # Sort best edges by similarity
        sorted_candidates = sorted(best_edges.items(), key=lambda x: x[1][0], reverse=True)

        # If edges not in same cluster, merge them
        merges_made = False
        for root_u, (score, root_v) in sorted_candidates:
            if find(root_u) != find(root_v):
                new_root, child1, child2 = union(root_u, root_v)
                if new_root is not None:
                    active_roots.discard(child1)
                    active_roots.discard(child2)
                    active_roots.add(new_root)
                    merges_made = True

        if not merges_made:
            break

    # Find the roots of the HAC tree
    final_roots = [r for r in active_roots if parent[r] == r]

    # Create a lookup table to know which node is the parent node
    tree_parent = {}
    for p, childs in children.items():
        for c in childs:
            tree_parent[c] = p

    # Generate the token paths (DFS tokenization)
    paths = []
    for v in range(num_nodes):
        path = [v]
        cur = v

        # Climb up until we reach a root
        while cur in tree_parent:
            cur = tree_parent[cur]
            path.append(cur)

        # Reverse to get the root in first position
        path.reverse()
        paths.append(torch.tensor(path, dtype=torch.long))

    return paths, final_roots


def preprocess_mnist(dataset):
    """Preprocess MNIST dataset with normalization, augmentation, and ordering."""
    processed = []
    print("Preprocessing: Normalizing, Augmenting (Pos), and Ordering...")

    for i in tqdm(range(len(dataset))):
        data = dataset[i].clone()

        # Pixel normalization
        if data.x.max() > 1.0:
            data.x = data.x / 255.0

        # Spatial info
        pos_mean = data.pos.mean(dim=0, keepdim=True)
        pos_max = data.pos.abs().max()
        if pos_max > 0:
            data.pos = (data.pos - pos_mean) / pos_max

        # Concat pos as features
        data.x = torch.cat([data.x, data.pos], dim=-1)

        # HAC (DFS)
        paths, roots = hac_dfs_tokenization(data)

        # Sort node indices by their [root, ..., leaf] path
        path_tuples = [tuple(p.tolist()) for p in paths]
        perm_list = sorted(range(len(path_tuples)), key=lambda idx: path_tuples[idx])
        perm = torch.tensor(perm_list, dtype=torch.long)

        # Reorder node features and positions
        data.x = data.x[perm]
        data.pos = data.pos[perm]

        # Reorder edge_index
        inv_perm = torch.zeros_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), dtype=torch.long)
        data.edge_index = inv_perm[data.edge_index]

        processed.append(data)

    return processed


def preprocess_color_connectivity(dataset):
    """Preprocess Color Connectivity dataset with HAC-DFS ordering."""
    processed = []
    print("Preprocessing: Applying HAC-DFS tokenization...")

    for i in tqdm(range(len(dataset))):
        data = dataset[i].clone()

        # HAC (DFS)
        paths, roots = hac_dfs_tokenization(data)

        # Sort node indices by their [root, ..., leaf] path
        path_tuples = [tuple(p.tolist()) for p in paths]
        perm_list = sorted(range(len(path_tuples)), key=lambda idx: path_tuples[idx])
        perm = torch.tensor(perm_list, dtype=torch.long)

        # Reorder node features
        data.x = data.x[perm]

        # If pos is separate, reorder it too
        if hasattr(data, 'pos') and data.pos is not None:
            data.pos = data.pos[perm]

        # Reorder edges
        inv_perm = torch.zeros_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), dtype=torch.long)
        data.edge_index = inv_perm[data.edge_index]

        processed.append(data)

    return processed
