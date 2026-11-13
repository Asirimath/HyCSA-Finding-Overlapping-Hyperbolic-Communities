import cProfile
import pstats
import json
import time

import matplotlib.pyplot as plt
from scipy.sparse import load_npz,save_npz

from Hyperbolic_community_detection_utils_version_sets_SA_Sparse import *
#from Hyperbolic_community_detection_utils_version_sets import *

rng = np.random.default_rng(42)


def experiment_overlapping_comm(noise_level,mat_no_list, alpha, Initial_Seeding_Method, Scoring_method,bs, p_add, Alpha_temperature, Not_mutiple_memb_Para = False, max_cliqu_para = 0, Save_results=False, Community_Expansion_Method='SA'):


    for mat_no in mat_no_list:

        print( "Matrix No", mat_no)
        A_noise = load_npz(f"Synthetic_Data/Adj_matrices_Overlaping_Comm/Overlapping_Comm_varing_sizes/adj_noise_{noise_level}_shape_Normal_mat_{mat_no}.npz")
        graphs_info_full = np.load(
          f"Synthetic_Data/Adj_matrices_Overlaping_Comm/Overlapping_Comm_varing_sizes/Community_info/adj_noise_{noise_level}_shape_Normal_mat_{mat_no}.npy", allow_pickle=True).item()
        labels_list=np.load(f"Synthetic_Data/Adj_matrices_Overlaping_Comm/Overlapping_Comm_varing_sizes/Vertex_label_info/adj_noise_{noise_level}_shape_Normal_label_info_mat_{mat_no}.npy")
        vert_list=np.load(f"Synthetic_Data/Adj_matrices_Overlaping_Comm/Overlapping_Comm_varing_sizes/Vertex_label_info/adj_noise_{noise_level}_shape_Normal_vertex_info_mat_{mat_no}.npy")

        # Seeding methods
        if Initial_Seeding_Method == 'sample_topk_percent_per_label':
            Initial_Seeding_Method_Name = 'topk'
            seeding_param = 0.1 #0.10
        elif Initial_Seeding_Method == 'short_random_walk2':
            Initial_Seeding_Method_Name = 'srw2'
            no_comm=len(np.unique(labels_list))
            seeding_param = [int(4*no_comm*np.log(no_comm)),50,1000,0.1, 0]  # [num_seeds, max_walk_length, num_walks from the vertex, pick top_k vertices, #of communities]
            seeding_param[4]= no_comm

        ##Run the main program####
        A_noise.sum_duplicates()
        A_noise.data[:] = 1
        A_noise.eliminate_zeros()
        A_noise.sort_indices()

        A_temp, F1_score, Recall_score, Precis_score, F1J_score, RecallJ_score, PrecisJ_score, True_communities, Pred_communities, Pred_communities_louvain, Subgraph_list, f1_louvain, recall_louvain, precision_louvain, precisionJ_louvain, recallJ_louvain, f1J_louvain, No_communities, Running_time, Community_dict = (
            run_hyperbolic_community_detection_algo_SA_Sparse(A_noise, labels_list, vert_list, max_cliqu_para, alpha, Initial_Seeding_Method, seeding_param, p_add,Alpha_temperature, Not_mutiple_memb_Para, Scoring_method, bs, Return_Para=True))

        A_temp_sparse = csr_matrix(A_temp.astype(np.uint8))
        A_temp_sparse.sum_duplicates()
        A_temp_sparse.data[:] = 1
        A_temp_sparse.eliminate_zeros()
        A_temp_sparse.sort_indices()

        F_log_likelihood_Eig_top_sorted=np.array([F1_score,Recall_score,Precis_score])
        F1J_log_likelihood_Eig_top_sorted=np.array([F1J_score,RecallJ_score,PrecisJ_score])

        F1_log_likelihood_louvain = np.array([f1_louvain, recall_louvain, precision_louvain])
        F1J_log_likelihood_louvain = np.array([f1J_louvain, recallJ_louvain, precisionJ_louvain])

        True_Comm_Eig_top_sorted = np.array(True_communities)
        Pred_Comm_Eig_top_sorted = np.array(Pred_communities)
        Pred_Comm_Louvain_Eig_top_sorted = np.array(Pred_communities_louvain)

        if Save_results:

            save_npz(f"Results/Synthetic_Data_Results/Results_Overlaping_Comm/Adj_matrices/adj_noise_{noise_level}_shape_Normal_mat_{mat_no}.npz", A_temp_sparse)

            np.save(f"Results/Synthetic_Data_Results/Results_Overlaping_Comm/Stat_Scores/Our_method/ll_Eig_{noise_level}_shape_Normal_mat_no_{mat_no}_SM_{Initial_Seeding_Method_Name}.npy", F_log_likelihood_Eig_top_sorted)
            np.save(f"Results/Synthetic_Data_Results/Results_Overlaping_Comm/Stat_Scores/Our_method/ll_EigJ_{noise_level}_shape_Normal_mat_no_{mat_no}_SM_{Initial_Seeding_Method_Name}.npy", F1J_log_likelihood_Eig_top_sorted)

            np.save(f"Results/Synthetic_Data_Results/Results_Overlaping_Comm/Stat_Scores/Louvain/ll_Louvain_{noise_level}_shape_Normal_mat_no_{mat_no}.npy", F1_log_likelihood_louvain)
            np.save(f"Results/Synthetic_Data_Results/Results_Overlaping_Comm/Stat_Scores/Louvain/ll_LouvainJ_{noise_level}_shape_Normal_mat_no_{mat_no}.npy", F1J_log_likelihood_louvain)

            np.save(f"Results/Synthetic_Data_Results/Results_Overlaping_Comm/True_Communities/True_Comm_{noise_level}_shape_Normal_mat_no_{mat_no}.npy", True_Comm_Eig_top_sorted)
            np.save(f"Results/Synthetic_Data_Results/Results_Overlaping_Comm/Pred_Communities/Our_method/Pred_Comm_{noise_level}_shape_Normal_mat_no_{mat_no}_SM_{Initial_Seeding_Method_Name}.npy", Pred_Comm_Eig_top_sorted)
            np.save(f"Results/Synthetic_Data_Results/Results_Overlaping_Comm/Pred_Communities/Louvain/Pred_Comm_{noise_level}_shape_Normal_mat_no_{mat_no}.npy",Pred_Comm_Louvain_Eig_top_sorted)

            np.save(f"Results/Synthetic_Data_Results/Results_Overlaping_Comm/Running_Time/Run_time_Eig_top_{noise_level}_shape_Normal_mat_no_{mat_no}_SM_{Initial_Seeding_Method_Name}.npy", Running_time)
            np.save(f"Results/Synthetic_Data_Results/Results_Overlaping_Comm/Pred_Communities/Our_method/Dict_Eig_top_{noise_level}_shape_Normal_mat_no_{mat_no}_SM_{Initial_Seeding_Method_Name}.npy", Community_dict)

    return 0





if __name__ == "__main__":

    #with cProfile.Profile() as prof:

       # ----------------#
       # To run Overlapping#
        start_time = time.time()
        mat_no_list = [0,1,2,3,4,5,6,7,8,9]
        noise_level = 0.01 #0.01,0.0,0.04
        Not_mutiple_memb_Para = False  # False for overlapping case., not a used parameter
        alpha=0.7 #We do not need this
        max_cliqu_para=0 #0 for all communities

        bs=10 # we used bs=10
        p_add = 0.8  # probality of sampling
        Scoring_method = 'Score' #No_Score
        Initial_Seeding_Method = 'sample_topk_percent_per_label' # sample_topk_percent_per_label, short_random_walk2
        Save_results_temp= False
        Alpha_temperature = 1
        t=experiment_overlapping_comm(noise_level, mat_no_list, alpha, Initial_Seeding_Method,  Scoring_method, bs, p_add, Alpha_temperature, Not_mutiple_memb_Para=False, max_cliqu_para=0, Save_results=Save_results_temp)
        end_time = time.time()
        print("Total time: ", end_time - start_time)
       # # ----------------------------#

        print("Done")

        # results= pstats.Stats(prof)
        # results.sort_stats(pstats.SortKey.TIME).print_stats()
