import time

from ac_fast_wrapper_sparse import check_openmp_threads,compute_AC_grid_cpp_local_overlap,build_Ac_matrix_sparse_with_counts_cpp

from Support_files_Sparse import *
from Support_files_2_Sparse import *


from maxmatchstat import *

import networkx as nx
import random as rnd
from scipy.sparse import save_npz


rnd.seed(0)
np.random.seed(0)


def Compute_good_param_hyperbol_v2( community_dict,Subgraph_sub_list,Indx_temp):

    # gamma_para_best = np.zeros(len(Subgraph_sub_list))
    # H_para_best = np.zeros(len(Subgraph_sub_list))
    gamma_para_best=3 #3
    H_para_best=0

    max_val1 = len(Subgraph_sub_list)
    max_val2 = int(len(Subgraph_sub_list)-1) #int(len(Subgraph_sub_list)/2) #110
    log_like_initial = float('-inf')

    #A_sub_block = np.zeros((L, L))
    Subgraph_sub_list_unique = list(dict.fromkeys(Subgraph_sub_list))

    All_edges_community = community_dict[Indx_temp]['edges']
    A_sub_block_sort, global_sort_Indxes = community_degree_sorted_matrix(All_edges_community, Subgraph_sub_list_unique)

    for gamma_test in range(1,max_val1):
        for H_test in range(1,max_val2):

            if gamma_test > H_test:

                P1_check = A_sub_block_sort.shape[0] - 1 + H_test - 2 * gamma_test
                P2_check = gamma_test - (2*(A_sub_block_sort.shape[0]-1)*H_test)/(A_sub_block_sort.shape[0]-1+H_test)

                if P1_check > 0 and P2_check >= 0:
                #if P1_check > 0:
                    log_like_temp = compute_AC_fast(A_sub_block_sort, gamma_test,H_test, mul_comm=False)

                    if log_like_temp >= log_like_initial:
                        gamma_para_best = gamma_test
                        H_para_best = H_test
                        log_like_initial = log_like_temp
                        #print(gamma_test,H_test,log_like_temp)

    return gamma_para_best, H_para_best, log_like_initial


def compute_global_likelihood_sparse(Community_dict, C_ordered, N, A_edge_count, A_indptr_i64, A_indices_i64):

    eps = 1e-12

    # ============================================================
    # 1. Community inside likelihood contributions
    # ============================================================

    sum_local_ll = 0.0

    for ind in C_ordered:

        d_c = float(Community_dict[ind]['d_c'])

        Area_EC = int(Community_dict[ind]['Area_EC'])
        Area_EC_Comp = int( Community_dict[ind]['Area_EC_Comp'])

        if not (0.0 <= d_c <= 1.0):
            raise ValueError( f"Community {ind}: invalid density {d_c}")

        sum_local_ll += (Area_EC * np.log(d_c + eps) + Area_EC_Comp * np.log(1.0 - d_c + eps) )

    # ============================================================
    # 2. Union of all community A_C regions
    # ============================================================

    _, claimed_count, edges_inside_AC = build_Ac_matrix_sparse_with_counts_cpp( Community_dict, N, -1,'Ac_edges', A_indptr_i64, A_indices_i64)

    claimed_count = int(claimed_count)
    edges_inside_AC = int(edges_inside_AC)

    # ============================================================
    # 3. Outside-region counts
    #
    # Exclude all diagonal positions consistently.
    # ============================================================

    d_o1 = int(A_edge_count) - edges_inside_AC
    d_o2 = N * (N - 1) - claimed_count

    if d_o2 < 0 or not (0 <= d_o1 <= d_o2):
        raise ValueError( "Invalid outside-region edge or area counts")

    # ============================================================
    # 4. Global outside density
    # ============================================================

    if d_o2 > 0:
        d_o = float(d_o1) / float(d_o2)
    else:
        d_o = 0.0

    # ============================================================
    # 5. Outside likelihood
    # ============================================================

    outside_ll = d_o1 * np.log(d_o + eps)+ (d_o2 - d_o1) * np.log(1.0 - d_o + eps)

    # ============================================================
    # 6. Total global likelihood
    # ============================================================

    total_ll = sum_local_ll + outside_ll

    return total_ll, sum_local_ll, outside_ll, d_o



def Compute_good_param_hyperbol_v3_new( community_dict,Subgraph_sub_list, Indx_temp, Ac_masked_others_precomputed):

    # ---------------------------------------------------------
    # Search ranges
    # ---------------------------------------------------------

    max_val_gamma = community_dict[Indx_temp]['gamma_para'] + 20
    min_val_gamma = max(community_dict[Indx_temp]['gamma_para'] - 20,1)

    max_val_H = community_dict[Indx_temp]['H_para'] + 8
    min_val_H = max(community_dict[Indx_temp]['H_para'] - 8, 1)
    # ---------------------------------------------------------
    # Unique nodes
    # ---------------------------------------------------------
    Subgraph_sub_list_unique = list(dict.fromkeys(Subgraph_sub_list))

    # ---------------------------------------------------------
    # Sparse AC mask belonging to OTHER communities
    # ---------------------------------------------------------

    Ac_masked_others = Ac_masked_others_precomputed


    # ---------------------------------------------------------
    # Build LOCAL dense community adjacency matrix
    # ---------------------------------------------------------

    All_edges_community = community_dict[Indx_temp]['edges']
    A_sub_block_sort, global_sort_Indxes = community_degree_sorted_matrix( All_edges_community, Subgraph_sub_list_unique )
    # ---------------------------------------------------------
    # Extract ONLY local overlap block from sparse global mask
    #
    # global mask:
    #       N x N sparse
    #
    # local block:
    #       nc x nc dense uint8
    # ---------------------------------------------------------

    global_sort_Indxes = np.asarray(global_sort_Indxes, dtype=np.int64)
    overlap_local = Ac_masked_others[global_sort_Indxes][:, global_sort_Indxes].toarray().astype(np.uint8)
    #overlap_local = np.ascontiguousarray(overlap_local, dtype=np.uint8)

    # ---------------------------------------------------------
    # Make local adjacency matrix suitable for C++
    # ---------------------------------------------------------

    A_sub_block_sort = np.ascontiguousarray(A_sub_block_sort, dtype=np.uint8)


    # ---------------------------------------------------------
    # C++ / OpenMP gamma-H grid search
    #
    # Pass LOCAL overlap mask rather than global N x N mask
    # ---------------------------------------------------------

    gamma_para_best, H_para_best, log_like_initial, D_c, Area_EC, Area_EC_comp = compute_AC_grid_cpp_local_overlap(A_sub_block_sort, min_val_gamma, max_val_gamma,
        min_val_H, max_val_H, overlap_local )


    return gamma_para_best, H_para_best, log_like_initial, D_c, Area_EC, Area_EC_comp, Ac_masked_others



def Initial_compute_new_likelihoods(N, community_dict, Q_prime, q_max, Subgraph_Indx_list, sub_graph):

# This will return the initial new likelihood. (We assume that commmunities do not overlap)
    for i in range(len(Q_prime)):
        if Q_prime[i] == q_max:
            # Compute the new likelihood
            indx = i
            Ac_masked_others = build_Ac_matrix_fast_dict_bool(community_dict, N, indx, 'Ac_edges')

            #L_new = len(sub_graph.nodes())
            gamma_para_new_temp, H_para_new_temp, log_Like_local_update = Compute_good_param_hyperbol_v2(community_dict, Subgraph_Indx_list[i],indx)
            community_dict[indx]['gamma_para'] = gamma_para_new_temp
            community_dict[indx]['H_para'] = H_para_new_temp
            community_dict[indx]['Ac_edges'] = Local_Ac_set( Subgraph_Indx_list[i], gamma_para_new_temp, H_para_new_temp, indx, community_dict,Ac_masked_others)

            break

    return Q_prime, Subgraph_Indx_list, log_Like_local_update, indx, gamma_para_new_temp, H_para_new_temp


def internal_degree(G, v, Q_nodes):
    return len(set(G.neighbors(v)) & Q_nodes)


def Compute_Partial_Global_log_likehood_from_counts(A_edge_count, N, claimed_count, edges_inside_AC, d_c_current, Area_EC_current, Area_Ec_block_comp_current):

    eps = 1e-12

    partial1 = Area_EC_current * np.log(d_c_current + eps) + Area_Ec_block_comp_current * np.log(1.0 - d_c_current + eps)

    # Number of positions outside all AC regions
    partial2_2 = N * N - claimed_count

    # Number of graph adjacency entries outside all AC regions
    partial2_1 = A_edge_count - edges_inside_AC

    if partial2_2 > 0:
        d_o = partial2_1 / partial2_2
    else:
        d_o = 0.0

    partial2 = partial2_1 * np.log(d_o + eps) + (partial2_2 - partial2_1) * np.log(1.0 - d_o + eps)

    return partial1 + partial2



def Compute_Max_Likelihood_Vertex_v0(N, A_u8_sparse,A_indptr_i64,A_indices_i64 , G, adj, c_values, true_communities, community_dict, ordered_indices_local_ll, A_edge_count, Temperature=1, sampling='Score',
        BS=5, p_add=0.25):


    Community_Count = 0

    # ============================================================
    # LOOP OVER COMMUNITIES
    # ============================================================

    for Indx_temp in ordered_indices_local_ll:

        Community_Count += 1
        candidates_dict = c_values[Indx_temp]

        if len(candidates_dict) == 0:
            continue


        # ========================================================
        # Candidate sampling
        # ========================================================

        if sampling == 'Score':
            k = 50 #50
            sorted_candidates = sorted([ (v, s) for v, s in candidates_dict.items() if s > 0 ], key=lambda x: x[1], reverse=True)
            v_samples_temp = [ v for v, _ in sorted_candidates[:k] ]
            v_samples = set(v_samples_temp)

        elif sampling == 'No_Score':
            v_samples = list(candidates_dict.keys())
            k = len(v_samples)

        v_samples_set = set(v_samples)

        # ========================================================
        # Edges belonging to OTHER communities
        # ========================================================

        edges_other = set().union(*[ community_dict[j]['edges'] for j in range(len(community_dict)) if j != Indx_temp ])

        # ========================================================
        # Current community nodes
        # ========================================================

        nodes_current_community = list( dict.fromkeys( community_dict[Indx_temp]['nodes']))
        # ========================================================
        # Select ADD / REMOVE proposals
        # ========================================================

        Selected_Vertices_Sampling = []
        Selected_Vertices = []

        while len(Selected_Vertices) < BS:

            rnd_guess = rnd.random()
            if rnd_guess <= p_add and len(v_samples) > 0:
                Selected_Vertices_Sampling.append(rnd.choice([*v_samples]))

            elif rnd_guess > p_add and len(nodes_current_community) > 0:
                Selected_Vertices_Sampling.append( rnd.choice( [*nodes_current_community] ) )

            Selected_Vertices = list( set(Selected_Vertices_Sampling) )

        # ========================================================
        # NEW:
        # Build mask of OTHER communities AND calculate
        #
        #   claimed_count_others
        #   edges_inside_others
        #
        # in ONE C++ call.
        #
        # No separate compute_global_counts_cpp() scan.
        # ========================================================

        #Ac_masked_others_fixed, claimed_count_others, edges_inside_others = build_Ac_matrix_sparse_with_counts( A_u8_sparse, community_dict, N, Indx_temp,"Ac_edges")
        Ac_masked_others_fixed, claimed_count_others, edges_inside_others = build_Ac_matrix_sparse_with_counts_cpp(community_dict, N, Indx_temp, "Ac_edges", A_indptr_i64, A_indices_i64)


        # ========================================================
        # Current accepted GLOBAL counts
        #
        # OTHER communities are already represented by:
        #
        #   claimed_count_others
        #   edges_inside_others
        #
        # Now add current community's Ac_edges.
        # ========================================================

        claimed_current, inside_current = compute_global_counts_from_others_and_current( A_u8_sparse, Ac_masked_others_fixed, claimed_count_others, edges_inside_others,
            community_dict[Indx_temp]['Ac_edges'])

        # ========================================================
        # Current global likelihood
        # ========================================================

        ll_glob_comm = Compute_Partial_Global_log_likehood_from_counts( A_edge_count, N, claimed_current, inside_current, community_dict[Indx_temp]['d_c'],
                community_dict[Indx_temp]['Area_EC'], community_dict[Indx_temp]['Area_EC_Comp'] )

        # ========================================================
        # PROCESS SELECTED VERTICES
        # ========================================================

        for e in range(len(Selected_Vertices)):

            nodes_current = list( dict.fromkeys( community_dict[Indx_temp]['nodes'] ) )

            # ====================================================
            # Save accepted state for rollback
            # ====================================================

            nodes_local = community_dict[Indx_temp]['nodes'].copy()
            Ac_edges_local =  community_dict[Indx_temp]['Ac_edges'].copy()
            gamma_local = community_dict[Indx_temp]['gamma_para']
            H_local =  community_dict[Indx_temp]['H_para']
            edges_local = community_dict[Indx_temp]['edges'].copy()
            dc_local = community_dict[Indx_temp]['d_c']
            Area_EC_local = community_dict[Indx_temp]['Area_EC']
            Area_EC_Comp_local =  community_dict[Indx_temp]['Area_EC_Comp']
            ll_local =  community_dict[Indx_temp]['local_ll']

            picked_node = Selected_Vertices[e]

            # ====================================================
            # ADD
            # ====================================================

            if picked_node in v_samples_set:

                v = picked_node
                nodes_current.append(v)

                # ------------------------------------------------
                # Update edges
                # ------------------------------------------------

                All_edges_claimed_by_v_temp =  edges_of_vertex( A_u8_sparse, nodes_current, v )
                All_edges_claimed_by_v = set(All_edges_claimed_by_v_temp)- edges_other
                community_dict[Indx_temp]['edges'] = community_dict[Indx_temp]['edges'] | All_edges_claimed_by_v
                # ------------------------------------------------
                # Recompute gamma / H
                # ------------------------------------------------

                community_dict[Indx_temp]['gamma_para'], community_dict[Indx_temp]['H_para'], ll_current, d_c_temp, Area_EC_temp, Area_Ec_block_comp_temp, Ac_masked_others = Compute_good_param_hyperbol_v3_new(
                    community_dict, nodes_current, Indx_temp, Ac_masked_others_precomputed= Ac_masked_others_fixed)

                # ------------------------------------------------
                # New AC region for current community
                # ------------------------------------------------

                Ac_edges_local_temp = Local_Ac_set_add_v_update( A_u8_sparse, nodes_current, community_dict[Indx_temp]['gamma_para'], community_dict[Indx_temp]['H_para'],
                        Indx_temp, community_dict, Ac_masked_others_fixed)


                # ------------------------------------------------
                # Store tentative state
                # ------------------------------------------------

                community_dict[Indx_temp]['d_c'] =  d_c_temp
                community_dict[Indx_temp]['Area_EC'] = Area_EC_temp
                community_dict[Indx_temp]['Area_EC_Comp'] =  Area_Ec_block_comp_temp
                community_dict[Indx_temp]['nodes'] = nodes_current.copy()
                community_dict[Indx_temp]['Ac_edges'] = Ac_edges_local_temp.copy()
                community_dict[Indx_temp]['labels'] = [Indx_temp] * len(nodes_current)
                community_dict[Indx_temp]['local_ll'] = ll_current

                # =================================================
                # Fast GLOBAL counts for proposed state
                # =================================================

                claimed_fast, inside_fast = compute_global_counts_from_others_and_current( A_u8_sparse, Ac_masked_others_fixed, claimed_count_others, edges_inside_others, Ac_edges_local_temp)


                # =================================================
                # Proposed GLOBAL likelihood
                # =================================================

                ll_glob_comm_temp = Compute_Partial_Global_log_likehood_from_counts(A_edge_count, N, claimed_fast, inside_fast, d_c_temp, Area_EC_temp, Area_Ec_block_comp_temp)

                # =================================================
                # Simulated annealing
                # =================================================

                delta = ll_glob_comm_temp- ll_glob_comm
                if delta >= 0:
                    accept = True

                else:
                    p = np.exp(delta / Temperature)
                    accept = np.random.rand() < p

                # =================================================
                # ACCEPT
                # =================================================

                if accept:
                    ll_glob_comm = ll_glob_comm_temp


                # =================================================
                # REJECT
                # =================================================

                else:
                    community_dict[Indx_temp]['nodes'] = nodes_local.copy()
                    community_dict[Indx_temp]['Ac_edges'] =  Ac_edges_local.copy()
                    community_dict[Indx_temp]['gamma_para'] = gamma_local
                    community_dict[Indx_temp]['H_para'] = H_local
                    community_dict[Indx_temp]['edges'] = edges_local.copy()
                    community_dict[Indx_temp]['d_c'] = dc_local
                    community_dict[Indx_temp]['labels'] =  [Indx_temp] * len(nodes_local)
                    community_dict[Indx_temp]['Area_EC'] =  Area_EC_local
                    community_dict[Indx_temp]['Area_EC_Comp'] = Area_EC_Comp_local
                    community_dict[Indx_temp]['local_ll'] = ll_local



            # ====================================================
            # REMOVE
            # ====================================================

            else:

                v = picked_node
                nodes_current.remove(v)
                # ------------------------------------------------
                # Remove its graph edges
                # ------------------------------------------------

                All_edges_claimed_by_v_temp =  edges_of_vertex( A_u8_sparse, nodes_current, v)

                All_edges_claimed_by_v = set(All_edges_claimed_by_v_temp)- edges_other
                community_dict[Indx_temp]['edges'] =  community_dict[Indx_temp]['edges'] - All_edges_claimed_by_v

                # ------------------------------------------------
                # Recompute gamma / H
                # ------------------------------------------------

                (community_dict[Indx_temp]['gamma_para'], community_dict[Indx_temp]['H_para'], ll_current, d_c_temp, Area_EC_temp,Area_Ec_block_comp_temp,
                 Ac_masked_others) = Compute_good_param_hyperbol_v3_new(community_dict, nodes_current, Indx_temp,Ac_masked_others_precomputed= Ac_masked_others_fixed)

                # ------------------------------------------------
                # New AC region
                # ------------------------------------------------
                Ac_edges_local_temp = Local_Ac_set_add_v_update(A_u8_sparse, nodes_current, community_dict[Indx_temp]['gamma_para'], community_dict[Indx_temp]['H_para'],
                                                                Indx_temp, community_dict, Ac_masked_others_fixed )


                # ------------------------------------------------
                # Store tentative state
                # ------------------------------------------------
                community_dict[Indx_temp]['d_c'] = d_c_temp
                community_dict[Indx_temp]['Area_EC'] =  Area_EC_temp
                community_dict[Indx_temp]['Area_EC_Comp'] = Area_Ec_block_comp_temp
                community_dict[Indx_temp]['nodes'] =  nodes_current.copy()
                community_dict[Indx_temp]['Ac_edges'] =  Ac_edges_local_temp.copy()
                community_dict[Indx_temp]['labels'] =  [Indx_temp] * len(nodes_current)
                community_dict[Indx_temp]['local_ll'] =  ll_current

                # =================================================
                # Fast GLOBAL counts
                # =================================================

                claimed_fast, inside_fast = compute_global_counts_from_others_and_current( A_u8_sparse, Ac_masked_others_fixed, claimed_count_others,
                                                                                           edges_inside_others, Ac_edges_local_temp)


                # =================================================
                # Proposed GLOBAL likelihood
                # =================================================

                ll_glob_comm_temp = Compute_Partial_Global_log_likehood_from_counts( A_edge_count, N, claimed_fast, inside_fast, d_c_temp, Area_EC_temp,
                        Area_Ec_block_comp_temp)

                # =================================================
                # Simulated annealing
                # =================================================
                delta = ll_glob_comm_temp - ll_glob_comm

                if delta >= 0:
                    accept = True

                else:
                    p = np.exp(delta / Temperature)
                    accept = np.random.rand() < p

                # =================================================
                # ACCEPT
                # =================================================

                if accept:

                    ll_glob_comm = ll_glob_comm_temp

                # =================================================
                # REJECT
                # =================================================

                else:
                    community_dict[Indx_temp]['nodes'] =  nodes_local.copy()
                    community_dict[Indx_temp]['Ac_edges'] = Ac_edges_local.copy()
                    community_dict[Indx_temp]['gamma_para'] =  gamma_local
                    community_dict[Indx_temp]['H_para'] = H_local
                    community_dict[Indx_temp]['edges'] = edges_local.copy()
                    community_dict[Indx_temp]['d_c'] =  dc_local
                    community_dict[Indx_temp]['labels'] =  [Indx_temp] * len(nodes_local)
                    community_dict[Indx_temp]['Area_EC'] = Area_EC_local
                    community_dict[Indx_temp]['Area_EC_Comp'] =  Area_EC_Comp_local
                    community_dict[Indx_temp]['local_ll'] = ll_local


# def initialize_edge_storage(N_global,Maximal_cliques):
#     claimed_global_mask = np.zeros((len(Maximal_cliques), N_global, N_global), dtype=bool)
#     Q_prime1_Ac_info2 = {}
#     return claimed_global_mask, Q_prime1_Ac_info2



def Compute_F1_Scores(true_communities, pred_communities):
    recall = best_match_recall(true_communities, pred_communities)
    precision = best_match_precision(true_communities, pred_communities)
    f1 = 2 * precision * recall / (precision + recall)

    return f1,recall,precision


def run_hyperbolic_community_detection_algo_SA_Sparse(A_Ori, Ori_lables, vert_set, No_comm_para, alpha, seeding_method, Seeding_param, p_add, Alpha_temperature, Not_mutiple_memb_para=False, Scoring_method='No_Score',batch_size_initial=10, Return_Para=False):

    """
    Parameters
    ----------
    A_Ori: Adj matrix of the graph
    Ori_lables : (List) List of original labels of the vertices e.g., [l0,l1,l1,...]
    vert_list : (List) List of vertices e.g., [v0,v0,v1,...]
    Not_mutiple_memb_para : (Binary) False is for the vertices can assign for multiple labels and True is for the vertices can assign for a single lable.
    Return_Para:

    Returns
    -------

    """
    check_openmp_threads()

    print("Simulation is running...")
    Start_time = time.monotonic()

    # A=A_Ori
    # A=A.astype(bool)
    A_u8_sparse = A_Ori.astype(np.uint8).tocsr()

    #-------------------------------------------
    A_u8_sparse.sum_duplicates()
    A_u8_sparse.data[:]=1
    A_u8_sparse.eliminate_zeros()
    A_u8_sparse.sort_indices()
    N = A_u8_sparse.shape[0]


    # Prepare C++ compatible CSR arrays ONCE
    A_indptr_i64 = np.ascontiguousarray( A_u8_sparse.indptr, dtype=np.int64)
    A_indices_i64 = np.ascontiguousarray( A_u8_sparse.indices, dtype=np.int64)
    #--------------------------------------

    print("Shape of A", A_u8_sparse.shape)
    print("Maximum # of Clusters", len(np.unique(np.array(Ori_lables))) )
    G = nx.from_scipy_sparse_array(A_u8_sparse, create_using=nx.Graph )
    # Total number of graph edges
    A_edge_count = A_u8_sparse.nnz

    # Getting Louvain Communities #
    true_communities = build_communities(vert_set, Ori_lables)
    pred_communities_louvain = nx.community.louvain_communities(G, seed=42)
    f1_louvain, recall_louvain, precision_louvain = Compute_F1_Scores(true_communities, pred_communities_louvain)
    precisionJ_louvain, recallJ_louvain, f1J_louvain = maximum_matching_stats(true_communities, pred_communities_louvain)
    print("Early F1_Louvain, recall_Louvain, precision_Louvain, len(louvain_comm)", f1_louvain, recall_louvain, precision_louvain,len(pred_communities_louvain))
    print("Early F1J_Louvain, recallJ_Louvain, precisionJ_Louvain",  f1J_louvain, recallJ_louvain, precisionJ_louvain)

   

    if seeding_method=='sample_topk_percent_per_label':
       Maximal_cliques_temp = sample_topk_percent_per_label(vert_set, Ori_lables, A_u8_sparse, percentage = Seeding_param)

    elif seeding_method == 'short_random_walk2':
        Maximal_cliques_temp1= collect_random_walk_seeds_precent_v0(G, num_seeds=Seeding_param[0], walk_length=Seeding_param[1], num_walks=Seeding_param[2], top_k=Seeding_param[3])
        Maximal_cliques_temp = select_random_walk_seeds_Jaccard(Maximal_cliques_temp1, n_select=Seeding_param[4])


    Maximal_cliques = [clq for clq in Maximal_cliques_temp if len(clq) >= 5]

    #Temprature for the SA and incremt para.
    Temp_Initial = 0.001 #Initial Temperature
    Temp = Temp_Initial
    alpha_temp =  Alpha_temperature#2,10, 25
    Temp_max= 25*alpha_temp #250


    batch_size=batch_size_initial
    alpha_batch_size=0 #5

    print(Maximal_cliques)
    print("# of initial seeds:", len(Maximal_cliques))

    # Stores the predicted labels and vertices respectively #
    Pre_lables = []
    Pre_vert = []

    # Q_prime1 stores the list of subgraph objects in an order, Subgraph_indx_list is a list of lists that stores the vertices that we selected [[Comm0],[[Comm2]],..]
    # Remove_vertex_list is a list that stores the selected vertices so far.
    Q_prime1 = []
    Subgraph_indx_list = []
    Remove_vertex_list = []
    community_dict={}

    for i in range(len(Maximal_cliques)):
        Q_prime1.append(G.subgraph(Maximal_cliques[i]).copy())
        Subgraph_indx_list.append(Maximal_cliques[i].copy())
        Pre_lables.extend([i] * len(Maximal_cliques[i]))
        Pre_vert.extend(Maximal_cliques[i].copy())

        Remove_vertex_list.extend(Maximal_cliques[i].copy())
        community_dict[i]= {
                "nodes": Maximal_cliques[i].copy(),
                "labels": [i] * len(Maximal_cliques[i]),
                "Ac_edges": set(),
                "edges": set(),
                "gamma_para":0,
                "H_para":0,
                 "d_c":0,
                 "Area_EC":0,
                 "Area_EC_Comp":0,
                 "local_ll":0
            }

    ####Computing early F1###
    pred_communities = build_communities(Pre_vert, Pre_lables)
    f1, recall, precision = Compute_F1_Scores(true_communities, pred_communities)
    print("Early F1,recall,precision", f1, recall, precision)
    #######################################


    Claimed_global_edge_set = build_edge_sets(A_u8_sparse, Subgraph_indx_list)
    for i in range(len(Claimed_global_edge_set)):
        community_dict[i]['edges']=Claimed_global_edge_set[i].copy()
        #print(len(community_dict[i]['edges']))

    print("Start doing initial ll")
    #Update the initial gamma, H, Ac information and community dict
    for i in range(len(Maximal_cliques)):
        _, _, community_dict[i]['local_ll'], _, _, _ = Initial_compute_new_likelihoods(N, community_dict, Q_prime1, Q_prime1[i], Subgraph_indx_list, Q_prime1[i])
    print("End doing initial ll")

    #Ordered community indices based on initial local log likelihoods#
    ordered_indices_local_ll = sorted(community_dict.keys(), key=lambda i: community_dict[i]["local_ll"], reverse=True)

    # Compute centrality values for each vertex...{0:cent0,1:cent1,...}
    adj = {u: set(G.neighbors(u)) for u in G}
    #centrality = nx.eigenvector_centrality(G, max_iter=3000)  # tol=1.0e-3

    count_nodes = len(Pre_vert) #0
    count_nodes_old=count_nodes

    Total_Sampling_Time=0
    Temp_Up_count=0

    print("Start doing SA")

    SA_running = True
    while SA_running:

        # ========================================================
        # Update temperature and batch size
        # ========================================================
        if count_nodes_old >= count_nodes:

            Temp += alpha_temp
            #batch_size += alpha_batch_size
            Temp_Up_count += 1

        else:

            if round(Temp, 4) > Temp_Initial:

                Temp -= alpha_temp
                # batch_size -= alpha_batch_size
                Temp_Up_count += 1

            else:

                Temp = Temp_Initial
                batch_size = batch_size_initial

        # Store current count for comparison in next iteration
        count_nodes_old = count_nodes

        # ========================================================
        # Compute candidate scores
        # ========================================================
        c_values = compute_c_values_NN_SA(Pre_vert, adj, community_dict, exclude_other_cliques=Not_mutiple_memb_para)

        # Communities having at least one candidate
        Temp_Q = [k for k, v in c_values.items() if len(v) != 0]

        # ========================================================
        # Check whether candidates exist
        # ========================================================
        if len(Temp_Q) == 0:

            print("len(Temp_Q) is zero")

            # Stop SA without using break
            SA_running = False

        else:

            # ====================================================
            # Run simulated annealing vertex update
            # ====================================================
            P_add = p_add

            Compute_Max_Likelihood_Vertex_v0( N, A_u8_sparse, A_indptr_i64, A_indices_i64, G, adj, c_values, true_communities,community_dict,
                                          ordered_indices_local_ll, A_edge_count, Temp, sampling=Scoring_method, BS=batch_size, p_add= P_add)

            # ====================================================
            # Reorder communities using current local likelihood
            # ====================================================
            ordered_indices_local_ll = sorted(community_dict.keys(), key=lambda i: community_dict[i]["local_ll"],reverse=True)

            # ====================================================
            # Update current community memberships
            # ====================================================
            Pre_vert = [v for indx in community_dict for v in community_dict[indx]["nodes"]]
            Pre_lables = [v for indx in community_dict for v in community_dict[indx]["labels"]]
            count_nodes = len(Pre_vert)

            # ====================================================
            # Stopping conditions
            # ====================================================
            if count_nodes >= len(vert_set): #len(vert_set)
                print("len(vert_set) is achieved")
                SA_running = False

            elif Temp >= Temp_max:
                print("Temp_max is achieved")
                SA_running = False

        pred_communities = build_communities(Pre_vert, Pre_lables)
        #TT = max_jaccard_matching(true_communities, pred_communities)
        f1, recall, precision = Compute_F1_Scores(true_communities, pred_communities)
        precisionJ, recallJ, f1J = maximum_matching_stats(true_communities, pred_communities)
        print("node count,f1,recall,precision, Not_mutiple_memb_para, Temp_Up_count", count_nodes, f1, recall, precision, Not_mutiple_memb_para,Temp_Up_count)
        print("node count, f1J, recallJ, precisionJ, sampling_time, len(pred[1])/len(true[1])", count_nodes,f1J, recallJ, precisionJ,Total_Sampling_Time, len(pred_communities[1])/len(true_communities[1]))
        print("# of initial seeds, len(vert_set), P_add Temperature, BS",len(Maximal_cliques), len(vert_set), P_add, Temp, batch_size )


    #Prediting communities
    pred_communities = build_communities(Pre_vert, Pre_lables)
    f1, recall, precision = Compute_F1_Scores(true_communities, pred_communities)
    precisionJ, recallJ, f1J = maximum_matching_stats(true_communities, pred_communities)
    Number_of_total_Pred_Elements = sum(len(s) for s in pred_communities)


    End_time= time.monotonic()
    running_time=End_time-Start_time

    print("Final F1,recall,precision", f1, recall, precision)
    print("Final F1J,recallJ,precisionJ", f1J, recallJ, precisionJ)
    print("Count_Nodes # of predited nodes", count_nodes,Number_of_total_Pred_Elements)
    print("Total time:", running_time)


    if Return_Para == False:
        return A_u8_sparse,f1,recall,precision,f1J,recallJ,precisionJ, true_communities, pred_communities,pred_communities_louvain, f1_louvain, recall_louvain, precision_louvain,precisionJ_louvain, recallJ_louvain, f1J_louvain, len(np.unique(np.array(Ori_lables))), running_time, community_dict

    else:
        return A_u8_sparse,f1,recall,precision,f1J,recallJ,precisionJ,  true_communities, pred_communities, pred_communities_louvain, Subgraph_indx_list, f1_louvain, recall_louvain, precision_louvain, precisionJ_louvain, recallJ_louvain, f1J_louvain, len(np.unique(np.array(Ori_lables))), running_time, community_dict
