import gzip
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse import coo_matrix, save_npz 


random.seed(2)


def load_filtered_communities(cmty_file, min_size=100, max_size=1000):
    unique_communities = []
    seen = set()

    with gzip.open(cmty_file, "rt") as f:
        for line in f:
            nodes = list(map(int, line.split()))

            if min_size <= len(nodes) <= max_size:
                community = tuple(sorted(nodes))

                if community not in seen:
                    seen.add(community)
                    unique_communities.append(list(community))

    return unique_communities

def remove_nested_communities(communities):

    communities_sets = [set(C) for C in communities]

    keep = []

    for i, C in enumerate(communities_sets):
        is_nested = False

        for j, D in enumerate(communities_sets):
            if i != j and C < D:   # C is a proper subset of D
                is_nested = True
                break

        if not is_nested:
            keep.append(communities[i])

    return keep



def sample_communities(communities, n_min=10, n_max=20):
    k = random.randint(n_min, n_max)
    return random.sample(communities, min(k, len(communities)))


def build_vertex_set(sampled_comms):
    vertex_set = set()
    for comm in sampled_comms:
        vertex_set.update(comm)

    return sorted(vertex_set)


def build_index_mapping(vertex_list):
    node_to_idx = {node: i for i, node in enumerate(vertex_list)}
    idx_to_node = {i: node for i, node in enumerate(vertex_list)}
    return node_to_idx, idx_to_node



def build_adj_matrix(edge_file, node_to_idx):

    N = len(node_to_idx)
    A = np.zeros((N, N), dtype=np.uint8)

    with gzip.open(edge_file, "rt") as f:
        for line in f:

            if line.startswith("#"):
                continue
            u, v = map(int, line.split())

            if u in node_to_idx and v in node_to_idx:

                i = node_to_idx[u]
                j = node_to_idx[v]

                A[i, j] = 1
                A[j, i] = 1

    return A



def build_adj_matrix_sparse(edge_file, node_to_idx):

    N = len(node_to_idx)

    rows = []
    cols = []

    with gzip.open(edge_file, "rt") as f:
        for line in f:

            if line.startswith("#"):
                continue

            u, v = map(int, line.split())

            if u in node_to_idx and v in node_to_idx:

                i = node_to_idx[u]
                j = node_to_idx[v]

                rows.append(i)
                cols.append(j)

                if i != j:
                    rows.append(j)
                    cols.append(i)

    data = np.ones(len(rows), dtype=np.int32 )

    A = coo_matrix( (data, (rows, cols)), shape=(N, N), dtype=np.int32).tocsr()

    A.sum_duplicates()

    # Convert all existing entries to 1
    A.data[:] = 1

    A = A.astype(np.uint8)

    A.eliminate_zeros()
    A.sort_indices()

    return A



def build_labels(sampled_comms, node_to_idx):

    N = len(node_to_idx)
    label_list = [[] for _ in range(N)]

    for comm_id, comm_nodes in enumerate(sampled_comms):
        for node in comm_nodes:
            if node in node_to_idx:
                idx = node_to_idx[node]
                label_list[idx].append(comm_id)

    return label_list


def build_sampled_dataset(cmty_file, edge_file,min_size,max_size):

    # ---- load communities ----
    communities_temp = load_filtered_communities(cmty_file,min_size,max_size)
    communities = remove_nested_communities(communities_temp)

    # ---- sample communities ----
    #sampled_comms = sample_communities(communities)

    # ---- build vertex list ----
    #vertex_list = build_vertex_set(sampled_comms)
    vertex_list = build_vertex_set(communities)

    # ---- build mappings ----
    node_to_idx, idx_to_node = build_index_mapping(vertex_list)

    # ---- build adjacency matrix ----
    A = build_adj_matrix_sparse(edge_file, node_to_idx)
    print("Size of A:",A.shape[0])
    
    # ---- build label list ----
    #label_list = build_labels(sampled_comms, node_to_idx)
    label_list = build_labels(communities, node_to_idx)

    return A, vertex_list, label_list, communities




cmty_file = "SNAP_Datasets/com-lj.top5000.cmty.txt.gz"
#cmty_file = "SNAP_Datasets/com-lj.all.cmty.txt.gz"
edge_file = "SNAP_Datasets/com-lj.ungraph.txt.gz"

for i in range(1):
    
    min_size=50 #300 from 100 we have 139 communities, 50
    max_size=1000 #850
    
    A, vertex_list, label_list, sampled_comms = build_sampled_dataset(cmty_file, edge_file,min_size,max_size)
    node_indices = []
    comm_indices = []
    
    
    print(len(sampled_comms))
    # for i in range(len(sampled_comms)):
    #     print('Length of the community:', len(sampled_comms[i]))
        
        
    for node_i, comm_list in enumerate(label_list):
        for q in comm_list:
            node_indices.append(node_i)
            comm_indices.append(q)

    node_indices = np.array(node_indices)
    comm_indices = np.array(comm_indices)

    r=0
    # Create a unique file name using parameters
    filename = f"LJ_adj_mat_{r}_{min_size}.npz"
    filepath = os.path.join("Adj_matrices/", filename)


    # Create community info files
    filename_comm_info_labels = f"Comm_Info_LJ_labels_{min_size}.npy"
    filename_comm_info_vert = f"Comm_Info_LJ_vert_{min_size}.npy"
    filepath__comm_info_labels = os.path.join("Adj_matrices/Community_Info/", filename_comm_info_labels)
    filepath__comm_info_verts = os.path.join("Adj_matrices/Community_Info/", filename_comm_info_vert)

    # np.save(filepath__comm_info_labels,comm_indices)
    # np.save(filepath__comm_info_verts,node_indices)
    
    # save_npz(filepath, A)

    # plt.figure()
    # plt.spy(A[0:1000,0:1000])

