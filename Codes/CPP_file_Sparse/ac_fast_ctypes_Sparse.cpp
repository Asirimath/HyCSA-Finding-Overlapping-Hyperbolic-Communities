#include <cmath>
#include <cstdint>
#include <limits>
#include <omp.h>
#include <cstdio>
#include <algorithm>
#include <unordered_set>
#include <cstddef>


#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif


// ============================================================
// TEST: Check how many OpenMP threads are actually being used
// ============================================================
EXPORT void check_openmp_threads()
{
    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("OpenMP threads actually running: %d\n",
                   omp_get_num_threads());

            printf("OpenMP max threads: %d\n",
                   omp_get_max_threads());
        }
    }
}





// ================================================================
// Evaluate ONE gamma,H combination
// ================================================================

static inline bool evaluate_one_ac_local_overlap(
    const uint8_t* A_sub,
    const uint8_t* overlap_local,
    int64_t nc,
    int gamma,
    int H,
    double& ll_out,
    double& dc_out,
    int64_t& area_ec_out,
    int64_t& area_ec_comp_out
)
{
    // ============================================================
    // Parameters defining A_C
    // ============================================================

    const double gamma_d = static_cast<double>(gamma);
    const double H_d = static_cast<double>(H);
    const double n_minus_1 = static_cast<double>(nc - 1);

    const double p0 = gamma_d * gamma_d- n_minus_1 * H_d;
    const double p1 =n_minus_1+ H_d - 2.0 * gamma_d;
    const double p2 = (gamma_d - H_d)* (gamma_d - H_d);
    const double p3 =(gamma_d - n_minus_1)* (gamma_d - n_minus_1);


    // SAME condition as dense evaluator
    if (std::abs(p1) < 1e-12)
    {
        return false;
    }


    const double alpha = p0 / p1;
    const double RHS =(p2 * p3)/ (p1 * p1);

    // ============================================================
    // Counters inside A_C
    // ============================================================

    int64_t Area_AC = 0;
    int64_t Area_EC = 0;
    int64_t Area_Ec_comp = 0;


    // ============================================================
    // Counters outside A_C
    // ============================================================

    int64_t Area_AC_comp = 0;
    int64_t Area_AC_comp2 = 0;
    int64_t K2_sum = 0;

    // ============================================================
    // SAME upper-triangle traversal as dense evaluator
    // ============================================================

    for (int64_t i = 0; i < nc; ++i)
    {
        const double i_alpha = static_cast<double>(i) + alpha;
        const int64_t local_row =i * nc;


        for (int64_t j = i + 1; j < nc; ++j) //for (int64_t j = i; j < nc; ++j)
        {
            const int64_t local_pos = local_row + j;


            // ====================================================
            // A DIFFERENCE FROM DENSE VERSION:
            //
            // Dense:
            //
            // claimed_global_mask[
            //     global_idx[i] * N_global
            //     + global_idx[j]
            // ]
            //
            // Sparse/local:
            //
            // overlap_local[i,j]
            // ====================================================

            if (overlap_local[local_pos] != 0)
            {
                continue;
            }


            // ====================================================
            // SAME weighting as dense version
            // ====================================================

            const int64_t weight = (i == j) ? 1 : 2;

            // ====================================================
            // SAME hyperbolic condition
            // ====================================================

            const double lhs = i_alpha *(static_cast<double>(j) + alpha);
            const bool in_AC =(lhs <= RHS);

            // ====================================================
            // Edge / non-edge
            // ====================================================

            const bool is_edge =(A_sub[local_pos] != 0);

            // ====================================================
            // Inside A_C
            // ====================================================

            if (in_AC)
            {
                Area_AC += weight;

                if (is_edge)
                {
                    Area_EC += weight;
                }
                else
                {
                    Area_Ec_comp += weight;
                }
            }

            // ====================================================
            // Outside A_C
            // ====================================================

            else
            {
                Area_AC_comp += weight;

                if (is_edge)
                {
                    Area_AC_comp2 += weight;
                }
                else
                {
                    K2_sum += weight;
                }
            }
        }
    }


    // ============================================================
    // Density inside A_C
    // ============================================================
    const double d_c = (Area_AC > 0) ? static_cast<double>(Area_EC) / static_cast<double>(Area_AC) : 0.0;

    // ============================================================
    // Density outside A_C
    // ============================================================

    const double d_o = (Area_AC_comp > 0) ? static_cast<double>(Area_AC_comp2) / static_cast<double>(Area_AC_comp) : 0.0;

    // ============================================================
    // SAME likelihood as dense evaluator
    // ============================================================

    constexpr double eps = 1e-12;

    const double ll = static_cast<double>(Area_EC)* std::log(d_c + eps) + static_cast<double>(Area_Ec_comp)* std::log(1.0 - d_c + eps) +
        static_cast<double>(Area_AC_comp2)* std::log(d_o + eps)+  static_cast<double>(K2_sum)* std::log(1.0 - d_o + eps);


    // ============================================================
    // Outputs
    // ============================================================

    ll_out = ll;
    dc_out = d_c;
    area_ec_out = Area_EC;
    area_ec_comp_out =Area_Ec_comp;


    return true;
}


// ================================================================
// OpenMP gamma/H grid search
// ================================================================
EXPORT int compute_ac_grid_local_overlap(
    const uint8_t* A_sub,
    const uint8_t* overlap_local,

    int64_t nc,

    int min_gamma,
    int max_gamma,

    int min_H,
    int max_H,

    int* best_gamma_out,
    int* best_H_out,

    double* best_ll_out,
    double* dc_out,

    int64_t* area_ec_out,
    int64_t* area_ec_comp_out
)
{
    if (!A_sub || !overlap_local)
    {
        return -1;
    }

    // --------------------------------------------------------
    // SAME initialization as dense implementation
    // --------------------------------------------------------

    double global_best_ll =-std::numeric_limits<double>::infinity();
    int global_best_gamma = 1;
    int global_best_H = 0;
    double global_best_dc = 0.0;
    int64_t global_best_area_ec = 0;
    int64_t global_best_area_ec_comp = 0;


    // ========================================================
    // OpenMP
    // ========================================================

    #pragma omp parallel
    {
        double local_best_ll =-std::numeric_limits<double>::infinity();

        int local_best_gamma = 1;
        int local_best_H = 0;
        double local_best_dc = 0.0;

        int64_t local_best_area_ec = 0;
        int64_t local_best_area_ec_comp = 0;


        // SAME search structure as dense implementation
        #pragma omp for collapse(2) schedule(dynamic, 4)
        for (int gamma_test = min_gamma; gamma_test < max_gamma; ++gamma_test)
        {
            for (int H_test = min_H;H_test < max_H; ++H_test)
            {

                // =================================================
                // SAME gamma/H condition as dense implementation
                //
                // Dense:
                // if (gamma_test <= H_test)
                //     continue;
                //
                // Therefore gamma must be STRICTLY greater than H.
                // =================================================

                if (gamma_test <= H_test)
                    continue;

                // =================================================
                // SAME P1 check as dense implementation
                // =================================================

                const double P1_check = static_cast<double>(nc - 1)+ H_test- 2.0 * gamma_test;


                // =================================================
                // SAME denominator check
                // =================================================

                const double denom =static_cast<double>(nc - 1)+ H_test;

                if (std::abs(denom) < 1e-12)
                    continue;


                // =================================================
                // SAME P2 check
                // =================================================

                const double P2_check =gamma_test-(2.0 *static_cast<double>(nc - 1) *H_test / denom);

                if (P1_check <= 0.0 ||
                    P2_check < 0.0)
                {
                    continue;
                }


                // =================================================
                // Evaluate exactly as dense version
                //
                // ONLY difference:
                // dense uses global claimed mask
                // sparse uses local overlap mask.
                // =================================================

                double ll = 0.0;
                double dc = 0.0;

                int64_t Area_EC = 0;
                int64_t Area_EC_comp = 0;

                const bool valid = evaluate_one_ac_local_overlap( A_sub, overlap_local, nc, gamma_test, H_test, ll, dc, Area_EC, Area_EC_comp);

                if (!valid)
                    continue;


                // =================================================
                // SAME update rule as dense implementation
                // =================================================

                if (ll >= local_best_ll)
                {
                    local_best_ll = ll;
                    local_best_gamma = gamma_test;
                    local_best_H = H_test;
                    local_best_dc = dc;
                    local_best_area_ec = Area_EC;
                    local_best_area_ec_comp =Area_EC_comp;
                }
            }
        }


        // ====================================================
        // SAME thread merge rule as dense implementation
        // ====================================================

        #pragma omp critical
        {
            if (local_best_ll >= global_best_ll)
            {
                global_best_ll = local_best_ll;
                global_best_gamma = local_best_gamma;
                global_best_H = local_best_H;
                global_best_dc = local_best_dc;
                global_best_area_ec = local_best_area_ec;
                global_best_area_ec_comp = local_best_area_ec_comp;
            }
        }
    }


    // --------------------------------------------------------
    // Outputs
    // --------------------------------------------------------

    *best_gamma_out = global_best_gamma;
    *best_H_out = global_best_H;
    *best_ll_out = global_best_ll;
    *dc_out = global_best_dc;

    *area_ec_out = global_best_area_ec;
    *area_ec_comp_out = global_best_area_ec_comp;


    return 0;
}




// ================================================================
// Check whether sparse CSR adjacency A contains edge (u,v)
//
// indptr:
//     row u begins at indptr[u]
//     row u ends   at indptr[u+1]
//
// indices:
//     contains column indices.
//
// IMPORTANT:
// A.indices must be sorted.
// ================================================================

static inline bool csr_has_edge( const int64_t* indptr, const int64_t* indices,
    int64_t u, int64_t v)
{
    const int64_t start = indptr[u];
    const int64_t end   = indptr[u + 1];

    return std::binary_search(indices + start, indices + end, v);
}


// ================================================================
// Build sparse Boolean AC union coordinates + counts
//
// edge_pairs memory layout:
//
//   [u0,v0,u1,v1,u2,v2,...,]
//
// For every undirected input pair (u,v), this function creates:
//
//   (u,v)
//   (v,u)
//
// unless u == v.
//
// Duplicate positions caused by overlapping communities are counted
// only once.
//
// Outputs:
//
// rows_out / cols_out:
//      unique claimed matrix positions.
//
// claimed_count_out:
//      number of unique claimed matrix positions.
//
// edges_inside_out:
//      number of claimed positions that are also edges in A.
// ================================================================

EXPORT int build_ac_sparse_counts_cpp(
    const int64_t* edge_pairs, int64_t num_edges, const int64_t* A_indptr,
    const int64_t* A_indices, int64_t N, int64_t* rows_out, int64_t* cols_out,
    int64_t* claimed_count_out, int64_t* edges_inside_out)

{


    // ------------------------------------------------------------
    // Hash set containing encoded N x N matrix positions.
    //
    // key = u*N + v
    //
    // This gives Boolean-union:
    // inserting the same position repeatedly has no effect.
    // ------------------------------------------------------------

    std::unordered_set<uint64_t> claimed;

    if (num_edges > 0)
    {
        claimed.reserve(static_cast<std::size_t>(2 * num_edges));
    }


    // ------------------------------------------------------------
    // Build union of AC positions
    // ------------------------------------------------------------

    for (int64_t e = 0; e < num_edges; ++e)
    {
        const int64_t u = edge_pairs[2 * e];
        const int64_t v = edge_pairs[2 * e + 1];


        // Position (u,v)
        const uint64_t key_uv = static_cast<uint64_t>(u)*static_cast<uint64_t>(N) + static_cast<uint64_t>(v);
        claimed.insert(key_uv);

        // Symmetric position (v,u)
        if (u != v)
        {
            const uint64_t key_vu = static_cast<uint64_t>(v) * static_cast<uint64_t>(N)  + static_cast<uint64_t>(u);
            claimed.insert(key_vu);
        }
    }


    // ------------------------------------------------------------
    // Convert unique positions back to rows/columns
    // and count actual graph edges.
    // ------------------------------------------------------------

    int64_t count = 0;
    int64_t edges_inside = 0;

    for (const uint64_t key : claimed)
    {
        const int64_t u = static_cast<int64_t>(key / static_cast<uint64_t>(N));
        const int64_t v = static_cast<int64_t>(key % static_cast<uint64_t>(N));

        rows_out[count] = u;
        cols_out[count] = v;

        // Does sparse graph A contain (u,v)?
        if (csr_has_edge(A_indptr, A_indices, u, v))
        {
            ++edges_inside;
        }


        ++count;
    }


    // ------------------------------------------------------------
    // Outputs
    // ------------------------------------------------------------

    *claimed_count_out = count;
    *edges_inside_out = edges_inside;

    return 0;
}