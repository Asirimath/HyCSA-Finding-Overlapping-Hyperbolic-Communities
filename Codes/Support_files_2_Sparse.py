import numpy as np
from scipy.sparse import triu,coo_matrix, csr_matrix
from collections import defaultdict


def build_communities(vertices, labels):
    comms = defaultdict(set)
    for v, lbl in zip(vertices, labels):
        comms[lbl].add(v)
    return list(comms.values())


def best_match_precision(true_communities, pred_communities):
    precision_sum = 0.0

    for C_pred in pred_communities:
        best_overlap = 0.0

        for C_true in true_communities:
            overlap = len(C_true & C_pred) / len(C_pred)
            best_overlap = max(best_overlap, overlap)

        precision_sum += best_overlap

    return precision_sum / len(pred_communities)


def best_match_recall(true_communities, pred_communities):
    recall_sum = 0.0

    for C_true in true_communities:
        best_overlap = 0.0

        for C_pred in pred_communities:
            overlap = len(C_true & C_pred) / len(C_true)
            best_overlap = max(best_overlap, overlap)

        recall_sum += best_overlap

    return recall_sum / len(true_communities)


def build_edge_sets(A, communities):

    A = A.tocsr()
    edge_sets = []

    for C in communities:

        nodes = np.asarray(
            list(dict.fromkeys(C)),
            dtype=np.int64
        )

        community_edges = set()

        if len(nodes) < 2:
            edge_sets.append(community_edges)
            continue

        A_sub = A[nodes, :][:, nodes]
        A_upper = triu(A_sub, k=1, format="coo")

        for i_local, j_local in zip(
            A_upper.row, A_upper.col
        ):
            u = int(nodes[i_local])
            v = int(nodes[j_local])

            community_edges.add(
                (min(u, v), max(u, v))
            )

        edge_sets.append(community_edges)

    return edge_sets


def build_edge_sets_old(A, communities):

    # Make sure A is CSR
    A = A.tocsr()

    edge_owner = {}   # (u, v) -> community index
    edge_sets = [set() for _ in communities]

    for k, C in enumerate(communities):

        nodes = np.asarray(list(C), dtype=np.int64)

        if len(nodes) < 2:
            continue

        # --------------------------------------------------------
        # Sparse adjacency matrix restricted to this community
        # --------------------------------------------------------
        A_sub = A[nodes][:, nodes]

        # --------------------------------------------------------
        # Keep only upper triangle:
        # avoids seeing both (u,v) and (v,u)
        # and ignores diagonal/self-loops
        # --------------------------------------------------------
        A_upper = triu(A_sub, k=1, format="coo")

        # --------------------------------------------------------
        # A_upper.row / col are LOCAL indices inside A_sub.
        # Convert them back to GLOBAL graph indices.
        # --------------------------------------------------------
        for i_local, j_local in zip(A_upper.row, A_upper.col):

            u = int(nodes[i_local])
            v = int(nodes[j_local])

            edge = (u, v) if u < v else (v, u)

            # Assign edge only to first community that claims it
            if edge not in edge_owner:

                edge_owner[edge] = k
                edge_sets[k].add(edge)

    return edge_sets


def community_degree_sorted_matrix(edges, vertices):

    # Step 1: map vertices → indices
    vertices = list(set(vertices))
    n = len(vertices)
    v_to_idx = {v: i for i, v in enumerate(vertices)}

    # Step 2: build adjacency matrix
    A_comm = np.zeros((n, n), dtype=bool)

    for u, v in edges:
        if u in v_to_idx and v in v_to_idx:
            i, j = v_to_idx[u], v_to_idx[v]
            A_comm[i, j] = 1
            A_comm[j, i] = 1

    # Step 3: compute degrees
    degrees = np.sum(A_comm, axis=1)

    # Step 4: sort by degree (descending)
    order = np.argsort(-degrees)

    # Step 5: reorder matrix
    A_sorted = A_comm[np.ix_(order, order)]

    # Step 6: reorder vertex list (important!)
    sorted_vertices = [vertices[i] for i in order]

    return A_sorted, sorted_vertices


def build_Ac_matrix_fast_dict_bool(Community_dict, N, Indx, Info):

    rows = []
    cols = []

    # ------------------------------------------------------------
    # Collect AC positions from all communities except Indx
    # ------------------------------------------------------------
    for indx, data in Community_dict.items():
        if indx == Indx:
            continue

        s = data[Info]
        if not s:
            continue

        for u, v in s:
            # (u, v)
            rows.append(u)
            cols.append(v)

            # symmetric position (v, u)
            if u != v:
                rows.append(v)
                cols.append(u)

    # ------------------------------------------------------------
    # No claimed AC positions
    # ------------------------------------------------------------
    if not rows:
        return csr_matrix((N, N), dtype=bool )

    # ------------------------------------------------------------
    # Construct sparse matrix
    # ------------------------------------------------------------
    data = np.ones( len(rows), dtype=np.uint8 )
    edge_matrix = coo_matrix((data, (rows, cols)), shape=(N, N), dtype=np.uint8 ).tocsr()

    # Overlapping communities may generate duplicate positions
    edge_matrix.sum_duplicates()

    # We only care whether a position is claimed or not
    edge_matrix.data[:] = 1

    edge_matrix.eliminate_zeros()
    edge_matrix.sort_indices()

    # Convert values to Boolean
    edge_matrix = edge_matrix.astype(bool)

    return edge_matrix


def compute_AC_fast(A_block, gamma, H, mul_comm=True):
    nc = A_block.shape[0]

    # ---- Precompute constants ----
    p0 = gamma**2 - (nc - 1) * H
    p1 = (nc - 1) + H - 2 * gamma
    p2 = (gamma - H)**2
    p3 = (gamma - (nc - 1))**2

    alpha = p0 / p1
    RHS = (p2 * p3) / (p1**2)

    # ---- Vectorized Ac_block ----
    idx = np.arange(nc)
    LHS = (idx[:, None] + alpha) * (idx[None, :] + alpha)
    Ac_block = (LHS <= RHS).astype(int)

    # ---- Efficient counting (no extra matrices) ----
    A_bool = A_block.astype(bool)
    Ac_bool = Ac_block.astype(bool)

    Area_EC = np.count_nonzero(A_bool & Ac_bool)
    Area_AC = np.count_nonzero(Ac_bool)
    Area_Ec_block_comp = np.count_nonzero(Ac_bool & ~A_bool)

    if Area_AC == 0:
        Area_AC = 2  # safeguard

    d_c = Area_EC / Area_AC

    if mul_comm:
        return Ac_block, Area_EC, Area_Ec_block_comp, d_c

    # ---- Complement likelihood ----
    Ac_comp = ~Ac_bool

    Area_AC_comp = np.count_nonzero(Ac_comp)
    Area_AC_comp2 = np.count_nonzero(A_bool & Ac_comp)
    K2_sum = np.count_nonzero(Ac_comp & ~A_bool)

    d_o = Area_AC_comp2 / Area_AC_comp if Area_AC_comp > 0 else 0

    log_like = (Area_EC * np.log(d_c + 1e-12) + Area_Ec_block_comp * np.log(1 - d_c + 1e-12) +Area_AC_comp2 * np.log(d_o + 1e-12) + K2_sum * np.log(1 - d_o + 1e-12)
    )

    return log_like


def compute_AC_fast_overlap_opt_vectorized_for_sets_update(A_sub_block_sort_bool, gamma, H,
        Ac_claimed_global_mask_otheres, Global_sort_Indxes):

    # ---------------------------------------------------------
    # Local community size
    # ---------------------------------------------------------

    nc = A_sub_block_sort_bool.shape[0]


    # ---------------------------------------------------------
    # Hyperbolic parameters
    # ---------------------------------------------------------

    p0 = gamma ** 2 - (nc - 1) * H

    p1 = (nc - 1) + H - 2 * gamma

    if p1 == 0:
        p1 = 1e-12

    p2 = (gamma - H) ** 2
    p3 = (gamma - (nc - 1)) ** 2

    alpha = p0 / p1
    RHS = (p2 * p3) / (p1 ** 2)


    # ---------------------------------------------------------
    # Build LOCAL AC mask
    #
    # This is only nc x nc, so keeping it dense is fine.
    # ---------------------------------------------------------

    idx = np.arange(nc)

    LHS = (idx[:, None] + alpha)*(idx[None, :] + alpha)

    Ac_bool = (LHS <= RHS)


    # ---------------------------------------------------------
    # Convert global indices to NumPy array
    # ---------------------------------------------------------

    Global_sort_Indxes = np.asarray(Global_sort_Indxes, dtype=np.int64)

    # ---------------------------------------------------------
    # Extract ONLY the nc x nc block from sparse global AC mask
    #
    # Ac_claimed_global_mask_otheres remains sparse.
    # Only this small local block becomes dense.
    # ---------------------------------------------------------

    overlap_local = Ac_claimed_global_mask_otheres[Global_sort_Indxes][ :, Global_sort_Indxes].toarray().astype(bool)


    # ---------------------------------------------------------
    # Remove AC positions already claimed by other communities
    # ---------------------------------------------------------

    Ac_bool &= ~overlap_local


    # ---------------------------------------------------------
    # Keep only upper triangular positions
    # ---------------------------------------------------------

    rows, cols = np.where(np.triu(Ac_bool, 1))


    # ---------------------------------------------------------
    # Convert local positions back to global node indices
    # ---------------------------------------------------------

    AC_edges_Indxx = {((Global_sort_Indxes[i], Global_sort_Indxes[j]) if Global_sort_Indxes[i] < Global_sort_Indxes[j]
            else ( Global_sort_Indxes[j], Global_sort_Indxes[i]) ) for i, j in zip(rows, cols)
            }


    return AC_edges_Indxx


def compute_c_values_NN_SA( pred_vertices, adj, community_dict, exclude_other_cliques=False):

    # alpha = 0.7 #0.7
    all_clique_vertices = set(pred_vertices)
    c_values = {}

    for Q in community_dict:

        Qset = set(community_dict[Q]['nodes'])

        tau_Q = set().union(*(adj[u] for u in Qset))
        candidates = tau_Q - Qset

        if exclude_other_cliques:
            candidates -= all_clique_vertices - Qset

        c_values[Q] = {}

        Q_len = len(Qset)
        # Q_sum = sum(centrality[u] for u in Qset)

        for v in candidates:

            neighbors_in_Q = Qset & adj[v]
            inter_size = len(neighbors_in_Q)

            if inter_size == 0:
                continue

            # connectivity score
            c_value_local = inter_size / Q_len

            # centrality score
            base_score = c_value_local

            # ---- probabilistic penalty ----
            # k_v = vertex_membership_count.get(v, 0) #WE DONT NEED SUCH PENALTY
            # penalty = np.exp(-lambda_penalty * k_v)

            penalty=1
            c_values[Q][v] = base_score * penalty

    return c_values


def Local_Ac_set(Subgraph_list_local, gamma_para, H_para, INDXX, community_dict, Ac_masked_others):

    # ============================================================
    # 1. Use the same observed edges as parameter fitting
    # ============================================================

    All_edges_community = community_dict[INDXX]['edges']

    # ============================================================
    # 2. Construct degree-sorted adjacency matrix
    # ============================================================

    A_sub_block_sort, global_sort_Indxes = community_degree_sorted_matrix( All_edges_community,Subgraph_list_local)

    A_sub_block_sort_bool = A_sub_block_sort.astype(bool)

    # ============================================================
    # 3. Construct A_C using the fitted gamma/H
    #    and the previously claimed region mask
    # ============================================================

    AC_edges_Indxx = compute_AC_fast_overlap_opt_vectorized_for_sets_update(A_sub_block_sort_bool, gamma_para, H_para, Ac_masked_others,global_sort_Indxes)

    return AC_edges_Indxx



def Local_Ac_set_old( Subgraph_list_local, gamma_para, H_para,INDXX, community_dict, Ac_masked_others):


    #All_edge_claimed_by_v = edges_of_vertex(A,Subgraph_list_local, Subgraph_list_local[-1])

    other_edges = set().union(*[community_dict[j]['edges'] for j in range(len(community_dict)) if j != INDXX])
    All_edges_community=community_dict[INDXX]['edges'] - other_edges

    A_sub_block_sort, global_sort_Indxes = community_degree_sorted_matrix( All_edges_community, Subgraph_list_local  )
    A_sub_block_sort_bool  =  A_sub_block_sort.astype(bool)
    AC_edges_Indxx = compute_AC_fast_overlap_opt_vectorized_for_sets_update(A_sub_block_sort_bool, gamma_para, H_para, Ac_masked_others, global_sort_Indxes)

    return AC_edges_Indxx



def build_Ac_matrix_sparse_with_counts(A_u8, Community_dict, N, Indx, Info):

    rows = []
    cols = []

    # ============================================================
    # Collect AC positions from all OTHER communities
    # ============================================================

    for indx, data in Community_dict.items():

        if indx == Indx:
            continue

        s = data[Info]

        if not s:
            continue

        for u, v in s:

            # (u, v)
            rows.append(u)
            cols.append(v)

            # symmetric position (v, u)
            if u != v:
                rows.append(v)
                cols.append(u)
    # ============================================================
    # No AC positions
    # ============================================================

    if len(rows) == 0:

        Ac_mask = csr_matrix( (N, N), dtype=bool)
        claimed_count = 0
        edges_inside_AC = 0

        return  Ac_mask, claimed_count, edges_inside_AC



    # ============================================================
    # Sparse AC union mask
    # ============================================================

    data = np.ones(len(rows), dtype=np.uint8)
    Ac_mask = coo_matrix( (data, (rows, cols)), shape=(N, N), dtype=np.uint8).tocsr()


    # ============================================================
    # Handle duplicate / overlapping AC claims
    # ============================================================
    Ac_mask.sum_duplicates()

    # Boolean union:
    # regardless of how many communities claim a position,
    # its final value is simply 1.
    Ac_mask.data[:] = 1
    Ac_mask.eliminate_zeros()
    Ac_mask.sort_indices()
    Ac_mask = Ac_mask.astype(bool)


    # ============================================================
    # Number of claimed matrix positions
    # ============================================================

    claimed_count = int(Ac_mask.nnz)
    # ============================================================
    # Number of graph adjacency entries lying inside AC union
    #
    # Both A_u8 and Ac_mask remain sparse.
    # ============================================================
    edges_inside_AC = int(A_u8.multiply(Ac_mask).nnz)

    return Ac_mask, claimed_count, edges_inside_AC


def compute_global_counts_from_others_and_current(A_u8, Ac_masked_others_fixed, claimed_count_others, edges_inside_others, Ac_edges_current):

    claimed_count = int(claimed_count_others)
    edges_inside_AC = int(edges_inside_others)

    # Handles the case where both (u,v) and (v,u)
    # accidentally occur in Ac_edges_current.
    seen = set()

    for u, v in Ac_edges_current:

        # --------------------------------------------------------
        # Canonical undirected pair
        # --------------------------------------------------------
        if u <= v:
            pair = (u, v)
        else:
            pair = (v, u)

        if pair in seen:
            continue

        seen.add(pair)

        u, v = pair

        # --------------------------------------------------------
        # Sparse CSR lookup:
        # Is this pair already claimed by another community?
        #
        # Ac_masked_others_fixed is symmetric, so checking
        # (u,v) is sufficient.
        # --------------------------------------------------------
        if Ac_masked_others_fixed[u, v] != 0:
            continue

        # --------------------------------------------------------
        # Position (u,v) becomes newly claimed
        # --------------------------------------------------------
        claimed_count += 1

        if A_u8[u, v] != 0:
            edges_inside_AC += 1

        # --------------------------------------------------------
        # Symmetric position (v,u)
        # --------------------------------------------------------
        if u != v:
            claimed_count += 1

            if A_u8[v, u] != 0:
                edges_inside_AC += 1

    return claimed_count, edges_inside_AC



def edges_of_vertex(A, Subgraph_list_local, v):

    edges = []

    # Nodes that currently belong to the community
    subgraph_nodes = set(Subgraph_list_local)

    # ------------------------------------------------------------
    # Get neighbors of v directly from CSR structure
    # ------------------------------------------------------------
    start = A.indptr[v]
    end = A.indptr[v + 1]

    neighbors_v = A.indices[start:end]

    # ------------------------------------------------------------
    # Keep only neighbors that are also in the community
    # ------------------------------------------------------------
    for u in neighbors_v:

        u = int(u)

        if u == v:
            continue

        if u not in subgraph_nodes:
            continue

        # Store undirected edge in canonical order
        if u < v:
            edge = (u, v)
        else:
            edge = (v, u)

        edges.append(edge)

    return edges



def Local_Ac_set_add_v_update(A,Subgraph_list_local, gamma_para, H_para,INDXX, community_dict, Ac_masked_others):


    All_edges_community = community_dict[INDXX]['edges']
    A_sub_block_sort, global_sort_Indxes = community_degree_sorted_matrix( All_edges_community, Subgraph_list_local  )
    A_sub_block_sort_bool  =  A_sub_block_sort.astype(bool)

    # Ac_masked_others_bool= Ac_masked_others.astype(bool)
    # AC_edges_Indxx = compute_AC_fast_overlap_opt_vectorized_for_sets_update_bool(A_sub_block_sort_bool, gamma_para, H_para, Ac_masked_others_bool, global_sort_Indxes)

    AC_edges_Indxx = compute_AC_fast_overlap_opt_vectorized_for_sets_update(A_sub_block_sort_bool, gamma_para, H_para, Ac_masked_others, global_sort_Indxes)


    return AC_edges_Indxx