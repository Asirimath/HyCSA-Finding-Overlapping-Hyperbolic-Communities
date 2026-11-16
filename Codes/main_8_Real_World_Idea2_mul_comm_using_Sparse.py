from Hyperbolic_community_detection_utils_version_sets_SA_Sparse import * #log_likelihood_v2_single_new_v3_Idea2_muliple_comm_new_v1
from scipy.sparse import load_npz,save_npz

import time

start_time = time.time()

if __name__ == "__main__":
    # with cProfile.Profile() as prof:

        DBLP_max_cliqu_para=0 #47
        Youtube_max_cliqu_para= 0 #234
        LJ_max_cliqu_para= 0 #502
        friendster_cliqu_para = 0 #1567
        Amazon_max_cliqu_para= 0 #48
        Orkut_max_cliqu_para=0 #  4242


        Scoring_method = 'Score' #If 'Score' we run with tightest nearest neighbours, otherwise "No_Score" we run with the all nearest neighbours.
        Not_mutiple_memb_Para = False #False for overlapping case, True for non overlapping case.
        Data_base= 'LJ' #friendster, Orkut, Amazon
        min_size = 50 #it is 50.
        Save_Data_Para = False #True
        bs=10 #batch size
        p_add=0.8 #probality of sampling
        Alpha_temperature=10 #alpha temperature
        alpha=0.7 #parameter for eigen centrality and degree socres

        Initial_Seeding_Method = 'rw'  # rss, rw

    #Database
        if Data_base=='friendster':
           max_cliqu_para = friendster_cliqu_para
        elif Data_base=='Youtube':
            max_cliqu_para = Youtube_max_cliqu_para
        elif Data_base=='DBLP':
            max_cliqu_para = DBLP_max_cliqu_para
        elif Data_base=='LJ':
            max_cliqu_para= LJ_max_cliqu_para
        elif Data_base == 'Amazon':
            max_cliqu_para = Amazon_max_cliqu_para
        elif Data_base == 'Orkut':
            max_cliqu_para = Orkut_max_cliqu_para

        A = load_npz(f'Real_World_Data/Adj_matrices/{Data_base}_adj_mat_0_{min_size}.npz') #5
        Ori_lables=np.load(f'Real_World_Data/Adj_matrices/Community_Info/Comm_Info_{Data_base}_labels_{min_size}.npy').tolist() #5
        vert_list = np.load(f'Real_World_Data/Adj_matrices/Community_Info/Comm_Info_{Data_base}_vert_{min_size}.npy').tolist()

        # Seeding methods
        if Initial_Seeding_Method == 'rss':
            seeding_param = 0.1  # seeding_param is initial Sampling percentage 0 and 1 (for real data we have used 0.20 for real world data).  in our experments it is 0.20.
            Initial_Seeding_Method_Name = 'topk'
        elif Initial_Seeding_Method == 'rw':
            T_len=len(np.unique(Ori_lables))
            seeding_param = [int(2*T_len*np.log(T_len)), 150, 1200, 0.3, T_len]  # [num_seeds, max_walk_length, num_walks from the vertex, pick top_k vertices, #of communities]
            Initial_Seeding_Method_Name = 'srw2'


        ##Run the main program, Normal is not supported####
        (A_temp, F1_score, Recall_score, Precis_score, F1J_score, RecallJ_score, PrecisJ_score, True_communities, Pred_communities, Pred_communities_louvain,
             Subgraph_list, f1_louvain, recall_louvain, precision_louvain, precisionJ_louvain, recallJ_louvain, f1J_louvain, No_communities, Running_time, Community_dict) =  run_hyperbolic_community_detection_algo_SA_Sparse(A, Ori_lables, vert_list, max_cliqu_para, alpha, Initial_Seeding_Method, seeding_param, p_add, Alpha_temperature, Not_mutiple_memb_Para, Scoring_method, bs, Return_Para=True)

        F_log_likelihood_Eig_top_sorted=np.array([F1_score,Recall_score,Precis_score])
        F1J_log_likelihood_Eig_top_sorted=np.array([F1J_score,RecallJ_score,PrecisJ_score])

        F1_log_likelihood_louvain = np.array([f1_louvain, recall_louvain, precision_louvain])
        F1J_log_likelihood_louvain = np.array([f1J_louvain, recallJ_louvain, precisionJ_louvain])

        True_Comm_Eig_top_sorted = np.array(True_communities)
        Pred_Comm_Eig_top_sorted = np.array(Pred_communities)
        Pred_Comm_Louvain_Eig_top_sorted = np.array(Pred_communities_louvain)

        if Save_Data_Para:
            np.save(f"Results/Real_World_Data_Results/Stat_Scores/Our_method/likelihood_Eig_top_{Data_base}_Min_{min_size}_No_Comm_{No_communities}_SM_{Initial_Seeding_Method_Name}.npy", F_log_likelihood_Eig_top_sorted)
            np.save(f"Results/Real_World_Data_Results/Stat_Scores/Our_method/likelihood_Eig_topJ_{Data_base}_Min_{min_size}_No_Comm_{No_communities}_SM_{Initial_Seeding_Method_Name}.npy", F1J_log_likelihood_Eig_top_sorted)

            np.save(f"Results/Real_World_Data_Results/Stat_Scores/Louvain/likelihood_Louvain_{Data_base}_Min_{min_size}_No_Comm_{No_communities}.npy", F1_log_likelihood_louvain)
            np.save(f"Results/Real_World_Data_Results/Stat_Scores/Louvain/likelihood_LouvainJ_{Data_base}_Min_{min_size}_No_Comm_{No_communities}.npy", F1J_log_likelihood_louvain)

            np.save(f"Results/Real_World_Data_Results/True_Communities/True_Comm_Eig_top_{Data_base}_Min_{min_size}_No_Comm_{No_communities}.npy", True_Comm_Eig_top_sorted)
            np.save(f"Results/Real_World_Data_Results/Pred_Communities/Our_method/Pred_Comm_Eig_top_{Data_base}_Min_{min_size}_No_Comm_{No_communities}_SM_{Initial_Seeding_Method_Name}.npy", Pred_Comm_Eig_top_sorted)
            np.save(f"Results/Real_World_Data_Results/Pred_Communities/Louvain/Pred_Comm_Louvain_{Data_base}_Min_{min_size}_No_Comm_{No_communities}.npy",Pred_Comm_Louvain_Eig_top_sorted)

            np.save(f"Results/Real_World_Data_Results/Running_Time/Run_time_Eig_top_{Data_base}_Min_{min_size}_No_Comm_{No_communities}_SM_{Initial_Seeding_Method_Name}.npy", Running_time)
            np.save(f"Results/Real_World_Data_Results/Pred_Communities/Our_method/Dict_Eig_top_{Data_base}_Min_{min_size}_No_Comm_{No_communities}_SM_{Initial_Seeding_Method_Name}.npy", Community_dict)
            save_npz(f"Results/Real_World_Data_Results/Adj_Matrices/{Data_base}_adj_mat_Min_{min_size}_No_Comm_{No_communities}.npz", A_temp)