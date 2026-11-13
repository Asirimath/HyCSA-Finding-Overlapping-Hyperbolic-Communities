import numpy as np
import networkx as nx
import itertools
import matplotlib.pyplot as plt
import os
from scipy.sparse import csr_matrix, save_npz


np.random.seed(2)


def Arithmatic_Prog(n=10, a1=3, d=4):
    ap = a1 + d * np.arange(n)
    return ap

def hyperbolic_Commu(thresh, n=10, a1=2, d=4):#density

    AP = Arithmatic_Prog(n, a1, d)
    A_temp = np.outer(AP, AP)
    AA_temp = A_temp < thresh;
    A = AA_temp.astype(int)

    return A


def create_star(n):
    #n  # number of nodes
    G = nx.star_graph(n - 1)  # creates a star with n nodes (0 is center)
    A = nx.to_numpy_array(G, dtype=int)
    
    return A


def add_uniform_noise(A, p_noise):
    """
    Add symmetric uniform noise to an adjacency matrix A.
    Each edge/non-edge is flipped with probability p_noise.
    """
    n = A.shape[0]

    # Generate random mask of flips
    flips = np.random.rand(n, n) < p_noise

    # Apply XOR to flip edges (1->0, 0->1)
    A_noisy = np.logical_xor(A, flips).astype(int)

    # Ensure the matrix remains symmetric and no self-loops
    np.fill_diagonal(A_noisy, 0)
    A_noisy = np.triu(A_noisy, 1)
    A_noisy = A_noisy + A_noisy.T
    
    A_noisy = (A_noisy > 0).astype(float)
    
    
    return A_noisy



def generate_adj_mat_v2(Community_info, r, Thresh, Index_Points, N=1000):

    A = np.zeros((N, N), dtype=np.uint8)
    count = 0

    for comm in Community_info:

        row = Index_Points[count]
        col = Index_Points[count]

        size = Community_info[comm][1]
        comm_type = Community_info[comm][0]

        if comm_type == 'Normal':

            D = np.random.uniform(0.20, 0.30) #0.25,0.30
            A_local = hyperbolic_Commu( Thresh, size, a1=0, d=D )

        elif comm_type == 'Star':

            D = np.random.uniform(0.35, 0.45)
            A_local = hyperbolic_Commu(Thresh, size, a1=0, d=D)

        elif comm_type == 'Clique':

            D = np.random.uniform(0.08, 0.10)
            A_local = hyperbolic_Commu( Thresh, size, a1=0, d=D )

        else:
            count += 1
            continue


        # ============================================
        # UNION with already generated communities
        # Important when communities overlap
        # ============================================
        block = A[ row:row + size, col:col + size]
        A[row:row + size, col:col + size] = np.maximum(block, A_local)
        count += 1


    # Remove self-loops
    np.fill_diagonal(A, 0)

    return A


def Create_Community_Info(Index_Points,community_info):
    
    vertices = []
    labels = []

    for i, start in enumerate(Index_Points):
        size = community_info[i][1]
        end = start + size   # IMPORTANT
        
        for v in range(start, end):
            vertices.append(v)
            labels.append(i)
     
    return vertices, labels


def Create_Community_Info_v3(n_vertices,community_size,Community_shape_info,seed_info,max_comm_size,min_comm_size):
    
    Community_shape=['Normal','Star','Clique']
    #Community_shape=['Clique','Star','Clique']
  
    Indx=0
    Intersection_points=[] # Number of common points
    sizes=[]
    count=0
    Length=0  # Size of the adj matric
    Index_points=[] # List of starting points in the adj matrices
    
    while Length<=n_vertices:
        
        if count==0:
           Intersection_points.append(0)
           sizes_temp = np.random.randint(min_comm_size, max_comm_size) # Size of the community  (80 to 120)
           sizes_temp = np.maximum(sizes_temp, 10)  # ensure min 10 per community
           sizes.append( sizes_temp )
           Indx=0
           Index_points.append(Indx)
           Length=Indx+sizes[0]
           count+=1
                
        else:
            sizes_temp = np.random.randint(min_comm_size, max_comm_size) # Size of the community (min_comm_size to max_comm_size)
            sizes_temp = np.maximum(sizes_temp, 10)  # ensure min 10 per community
            sizes.append(sizes_temp)
            
            Intersection_points_temp=np.random.randint(10,40) # Number of common points (10 to 40)
            Intersection_points.append(Intersection_points_temp)
      
            Indx+=sizes[count-1]-Intersection_points[count]
            Index_points.append(Indx)
            Length=Indx+sizes[count]
            
            count+=1            
            #print(Indx)
    n_comms=count
    
    if Community_shape_info=='Vary':
        assigned_shapes =  np.random.choice(Community_shape, size=n_comms, replace=True)
    else:
        assigned_shapes=np.array([Community_shape[0]]*n_comms)
        
    community_info = {i: (shape, size) for i, (shape, size) in enumerate(zip(assigned_shapes, sizes))}
    
    
    return community_info,Index_points,Length



if __name__=="__main__": 
    
    # --- Parameter space ---
    n_vertices_approximate = 1500 #This can be change
    noise_levels = [0.0, 0.01, 0.02, 0.04, 0.08]
    Community_size=[20,20,40] #10
    Community_shape=['Normal','Star','Clique']
    Save_Results=False
    n_replicates = 10
    max_comm_size=151
    min_comm_size=50
    
    
    
    
    fixed = {'community_size':1, 'noise_levels': [], 'community_shape': 1}
    vary = {'community_size': [], 'noise_levels': 1, 'community_shape': []}
    
    fixed_keys = [k for k, v in fixed.items() if v != []]
    vary_keys = [k for k, v in vary.items() if v != []]
    
    
    C_shape_info= 'Vary' if 'community_shape' in vary_keys else 'Normal'
    matrices = []
    
    for C_size in Community_size if 'community_size' in vary_keys else [Community_size[0]]:
        for N_level in noise_levels if 'noise_levels' in vary_keys else [noise_levels[0]]:
            for r in range(n_replicates):
                
                if C_size==10:
                    Thresh= 20 #20
                elif C_size==20:
                    Thresh=15
                elif C_size==40:
                    Thresh=6
                
                community_info,Index_Points,size_A = Create_Community_Info_v3(n_vertices_approximate,C_size,C_shape_info,r,max_comm_size,min_comm_size)
                A = generate_adj_mat_v2(community_info, r,Thresh, Index_Points, size_A)
                A_noise=add_uniform_noise(A, N_level)
                Vert_list, Label_list = Create_Community_Info(Index_Points,community_info)
                
                
                A_noise_sparse = csr_matrix(A_noise.astype(np.uint8))
                A_sparse = csr_matrix(A.astype(np.uint8))

                A_noise_sparse.sum_duplicates()
                A_noise_sparse.data[:] = 1
                A_noise_sparse.eliminate_zeros()
                A_noise_sparse.sort_indices()
                
                
                A_sparse.sum_duplicates()
                A_sparse.data[:] = 1
                A_sparse.eliminate_zeros()
                A_sparse.sort_indices()
                
                
                plt.figure()
                plt.spy(A_noise)
                plt.grid()
                plt.show()
                
                matrices.append(A_noise)
                
                # Create a unique file name using parameters
                CC_size=len(community_info)
                filename_sparse = f"adj_noise_{N_level}_shape_{C_shape_info}_mat_{r}.npz"
                filepath_sparse = os.path.join(f"Adj_matrices_Overlaping_Comm\Overlapping_Comm_varing_sizes", filename_sparse)
                filename = f"adj_noise_{N_level}_shape_{C_shape_info}_mat_{r}.npy"
                filepath = os.path.join(f"Adj_matrices_Overlaping_Comm\Overlapping_Comm_varing_sizes", filename)
                
                
                # Create a unique file name using parameters for noise free
                filename_noise_free_sparse = f"adj_noise_{N_level}_shape_{C_shape_info}_mat_{r}.npz"
                filepath_noise_free_sparse = os.path.join(f"Adj_matrices_Overlaping_Comm/Overlapping_Comm_varing_sizes/Noise_free", filename_noise_free_sparse)
                filename_noise_free = f"adj_noise_{N_level}_shape_{C_shape_info}_mat_{r}.npy"
                filepath_noise_free = os.path.join(f"Adj_matrices_Overlaping_Comm/Overlapping_Comm_varing_sizes/Noise_free", filename_noise_free)
                
                
                # Create community info files
                filename_comm_info = f"adj_noise_{N_level}_shape_{C_shape_info}_mat_{r}.npy"
                filepath__comm_info = os.path.join(f"Adj_matrices_Overlaping_Comm/Overlapping_Comm_varing_sizes/Community_info", filename_comm_info)
                
                
                # Create community vertex list and label files
                filename_vertex_info = f"adj_noise_{N_level}_shape_{C_shape_info}_vertex_info_mat_{r}.npy"
                filepath__vertex_info = os.path.join(f"Adj_matrices_Overlaping_Comm/Overlapping_Comm_varing_sizes/Vertex_label_info", filename_vertex_info)
                filename_label_info = f"adj_noise_{N_level}_shape_{C_shape_info}_label_info_mat_{r}.npy"
                filepath__label_info = os.path.join(f"Adj_matrices_Overlaping_Comm/Overlapping_Comm_varing_sizes/Vertex_label_info", filename_label_info)
                
                
                
                # Save the matrix
                if Save_Results==True:
                    save_npz(filepath_sparse, A_noise_sparse)
                    save_npz(filepath_noise_free_sparse, A_sparse)
                    
                    np.save(filepath, A_noise)
                    np.save(filepath_noise_free, A)
                    np.save(filepath__comm_info,community_info)
                    np.save(filepath__vertex_info,Vert_list)
                    np.save(filepath__label_info,Label_list)
                    print(f"Saved {filepath}")
                
                #print(community_info)
         