import ctypes
import os
import platform


os.environ["OMP_NUM_THREADS"] = "16"

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.sparse import csr_matrix

# ============================================================
# Load DLL from the same directory as this Python file
# ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))

if platform.system() == "Windows":
    lib_path = os.path.join(base_dir, "ac_fast_sparse.dll")
    lib_path1 = os.path.join( os.path.dirname(__file__), "compute_AC_grid_cpp_global_do_sparse.dll" )

else:
    lib_path = os.path.join(base_dir, "ac_fast_sparse.so")
    lib_path1 = os.path.join( os.path.dirname(__file__), "compute_AC_grid_cpp_global_do_sparse.so" )

ac_lib = ctypes.CDLL(lib_path)
ac_global_do_lib = ctypes.CDLL(lib_path1)

ac_lib.check_openmp_threads.argtypes = []
ac_lib.check_openmp_threads.restype = None

ac_lib.compute_ac_grid_local_overlap.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64) ]

ac_lib.compute_ac_grid_local_overlap.restype = ctypes.c_int

ac_lib.build_ac_sparse_counts_cpp.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64) ]

ac_lib.build_ac_sparse_counts_cpp.restype = ctypes.c_int


ac_global_do_lib.compute_ac_grid_global_do_sparse.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,  ctypes.c_int,
ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.c_double, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)]

ac_global_do_lib.compute_ac_grid_global_do_sparse.restype = ctypes.c_int


# ============================================================
# Python wrapper
# ============================================================

def check_openmp_threads():
    ac_lib.check_openmp_threads()

def compute_AC_grid_cpp_local_overlap(A_sub_block_sort, min_gamma, max_gamma, min_H, max_H, overlap_local):

    # ------------------------------------------------------------
    # Local adjacency matrix
    # ------------------------------------------------------------
    A_sub = np.ascontiguousarray( A_sub_block_sort,dtype=np.uint8)

    # ------------------------------------------------------------
    # Local overlap mask
    # ------------------------------------------------------------
    overlap = np.ascontiguousarray(overlap_local,dtype=np.uint8)

    nc = A_sub.shape[0]
    # ------------------------------------------------------------
    # Outputs from C++
    # ------------------------------------------------------------

    best_gamma = ctypes.c_int()
    best_H = ctypes.c_int()

    best_ll = ctypes.c_double()
    d_c = ctypes.c_double()

    Area_EC = ctypes.c_int64()
    Area_EC_comp = ctypes.c_int64()


    # ------------------------------------------------------------
    # C++ call
    # ------------------------------------------------------------

    status = ac_lib.compute_ac_grid_local_overlap( A_sub.ctypes.data_as( ctypes.POINTER(ctypes.c_uint8)), overlap.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), ctypes.c_int64(nc),
        ctypes.c_int(min_gamma), ctypes.c_int(max_gamma), ctypes.c_int(min_H),ctypes.c_int(max_H), ctypes.byref(best_gamma), ctypes.byref(best_H), ctypes.byref(best_ll),
        ctypes.byref(d_c), ctypes.byref(Area_EC), ctypes.byref(Area_EC_comp))


    if status != 0:
        raise RuntimeError(f"compute_ac_grid_local_overlap failed with status={status}")


    return best_gamma.value, best_H.value, best_ll.value, d_c.value, Area_EC.value, Area_EC_comp.value




def build_Ac_matrix_sparse_with_counts_cpp( Community_dict, N, Indx, Info, A_indptr_i64, A_indices_i64):

    # ============================================================
    # Collect AC edges from all OTHER communities.
    # This part is still Python for now.
    # ============================================================

    edges = []

    for indx, data in Community_dict.items():

        if indx == Indx:
            continue

        s = data[Info]

        if not s:
            continue

        edges.extend(s)


    # ============================================================
    # No other AC edges
    # ============================================================

    if not edges:

        return (csr_matrix((N, N), dtype=bool), 0, 0)


    # ============================================================
    # One contiguous (M,2) array
    #
    # Memory:
    #
    # u0,v0,u1,v1,...
    # ============================================================

    edge_pairs = np.asarray(edges, dtype=np.int64)

    if edge_pairs.ndim != 2 or edge_pairs.shape[1] != 2:
        raise ValueError( f"Invalid edge array shape: {edge_pairs.shape}" )

    #edge_pairs = np.ascontiguousarray(edge_pairs, dtype=np.int64)
    num_edges = edge_pairs.shape[0]

    # ============================================================
    # Maximum possible number of output positions:
    #
    # Each input undirected edge can produce at most:
    #
    #   (u,v)
    #   (v,u)
    #
    # Therefore maximum = 2 * num_edges.
    #
    # Duplicates will result in a smaller claimed_count.
    # ============================================================

    max_positions = 2 * num_edges
    rows_out = np.empty( max_positions, dtype=np.int64)
    cols_out = np.empty( max_positions, dtype=np.int64)


    # ============================================================
    # C++ output scalars
    # ============================================================

    claimed_count = ctypes.c_int64()
    edges_inside_AC = ctypes.c_int64()

    # ============================================================
    # Call C++
    # ============================================================

    status = ac_lib.build_ac_sparse_counts_cpp( edge_pairs.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(num_edges), A_indptr_i64.ctypes.data_as( ctypes.POINTER(ctypes.c_int64)), A_indices_i64.ctypes.data_as( ctypes.POINTER(ctypes.c_int64) ),
        ctypes.c_int64(N), rows_out.ctypes.data_as( ctypes.POINTER(ctypes.c_int64)), cols_out.ctypes.data_as( ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(claimed_count), ctypes.byref(edges_inside_AC))


    if status != 0:
        raise RuntimeError("build_ac_sparse_counts_cpp "f"failed with status={status}")


    # ============================================================
    # Only first claimed_count positions are valid.
    # ============================================================

    count = claimed_count.value

    rows = rows_out[:count]
    cols = cols_out[:count]


    # ============================================================
    # Build sparse Boolean AC mask.
    #
    # C++ already removed duplicates, so every coordinate here
    # is unique.
    # ============================================================

    data = np.ones(count,dtype=bool)
    Ac_mask = csr_matrix((data, (rows, cols)), shape=(N, N), dtype=bool)
    Ac_mask.sort_indices()

    return Ac_mask, count, edges_inside_AC.value



def compute_AC_grid_cpp_global_do_sparse(A_sub_block_sort, min_gamma, max_gamma, min_H, max_H, global_sort_Indxes, M, d_o_global):


    # ============================================================
    # Local adjacency matrix
    # ============================================================

    A_sub = np.ascontiguousarray(A_sub_block_sort, dtype=np.uint8)
    nc = A_sub.shape[0]

    # ============================================================
    # Global indices corresponding to local sorted nodes
    # ============================================================

    global_idx = np.ascontiguousarray(global_sort_Indxes, dtype=np.int64)


    # ============================================================
    # Accumulated global mask M
    # ============================================================

    M_u8 = np.ascontiguousarray(M, dtype=np.uint8)
    N_global = M_u8.shape[0]


    # ============================================================
    # Global outside density
    # ============================================================

    d_o_global = float(d_o_global)


    # ============================================================
    # Output variables
    # ============================================================

    best_gamma = ctypes.c_int()
    best_H = ctypes.c_int()

    best_log_like = ctypes.c_double()
    best_d_c = ctypes.c_double()

    best_Area_EC = ctypes.c_int64()
    best_Area_Ec_comp = ctypes.c_int64()


    # ============================================================
    # Call C++
    # ============================================================

    status = ac_global_do_lib.compute_ac_grid_global_do_sparse(A_sub.ctypes.data_as( ctypes.POINTER(ctypes.c_uint8)), ctypes.c_int64(nc), ctypes.c_int(int(min_gamma)),
        ctypes.c_int(int(max_gamma)), ctypes.c_int(int(min_H)), ctypes.c_int(int(max_H)), global_idx.ctypes.data_as( ctypes.POINTER(ctypes.c_int64)),
        M_u8.ctypes.data_as( ctypes.POINTER(ctypes.c_uint8)), ctypes.c_int64(N_global), ctypes.c_double(d_o_global), ctypes.byref(best_gamma), ctypes.byref(best_H),
        ctypes.byref(best_log_like), ctypes.byref(best_d_c), ctypes.byref(best_Area_EC), ctypes.byref(best_Area_Ec_comp))


    # ============================================================
    # Check C++ return status
    # ============================================================

    if status == -1:
        raise RuntimeError(
            "compute_ac_grid_global_do failed: "
            "invalid pointer or matrix size."
        )

    if status == -2:
        raise RuntimeError(
            "compute_ac_grid_global_do failed: "
            "d_o_global must be in [0, 1]."
        )

    if status != 0:
        raise RuntimeError(
            f"compute_ac_grid_global_do failed "
            f"with status {status}."
        )


    # ============================================================
    # Make sure a valid gamma/H was found
    # ============================================================

    if not np.isfinite(best_log_like.value):
        raise RuntimeError(
            "No valid (gamma, H) pair was found."
        )


    # ============================================================
    # Return normal Python values
    # ============================================================

    return best_gamma.value, best_H.value, best_log_like.value, best_d_c.value, best_Area_EC.value, best_Area_Ec_comp.value
