#include <cmath>
#include <cstdint>
#include <limits>
#include <omp.h>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

EXPORT void set_openmp_threads(int n)
{
    if (n > 0)
    {
        omp_set_dynamic(0);
        omp_set_num_threads(n);
    }
}

struct ACGridResultGlobalDO
{
    double log_like;
    double d_c;
    int64_t Area_EC;
    int64_t Area_Ec_block_comp;
    bool valid;
};


static inline ACGridResultGlobalDO evaluate_one_ac_global_do(
    const uint8_t* A_sub,
    int64_t nc,
    double gamma,
    double H,
    const int64_t* global_idx,
    const uint8_t* M,
    int64_t N_global,
    double d_o_global)
{
    ACGridResultGlobalDO result;

    result.log_like =
        -std::numeric_limits<double>::infinity();

    result.d_c = 0.0;
    result.Area_EC = 0;
    result.Area_Ec_block_comp = 0;
    result.valid = false;

    // ============================================================
    // 1. Hyperbolic parameters
    // ============================================================

    const double n_minus_1 =static_cast<double>(nc - 1);

    const double p0 =gamma * gamma - n_minus_1 * H;

    const double p1 = n_minus_1 + H - 2.0 * gamma;

    const double p2 = (gamma - H) * (gamma - H);

    const double p3 = (gamma - n_minus_1) * (gamma - n_minus_1);

    if (std::abs(p1) < 1e-12){
        return result;
    }

    const double alpha = p0 / p1;
    const double RHS = (p2 * p3) / (p1 * p1);

    // ============================================================
    // 2. Counters inside A_C
    // ============================================================

    int64_t Area_AC = 0;
    int64_t Area_EC = 0;
    int64_t Area_Ec_comp = 0;

    // ============================================================
    // 3. Counters outside A_C
    // ============================================================

    int64_t Area_AC_comp = 0;
    int64_t Area_AC_comp2 = 0;
    int64_t K2_sum = 0;

    // ============================================================
    // 4. Evaluate hyperbolic region
    //
    // Exclude diagonal entries.
    // Each undirected pair contributes weight 2.
    // ============================================================

    for (int64_t i = 0; i < nc; ++i)
    {
        const int64_t gi = global_idx[i];

        const double i_alpha =static_cast<double>(i) + alpha;

        const int64_t local_row = i * nc;
        const int64_t global_row = gi * N_global;

        for (int64_t j = i + 1; j < nc; ++j)
        {
            const int64_t gj = global_idx[j];

            const int64_t local_pos = local_row + j;

            const int64_t global_pos = global_row + gj;

            // Exclude previously claimed positions
            if (M[global_pos] != 0)
                continue;

            // Count both symmetric matrix positions
            const int64_t weight = 2;

            // Hyperbolic region condition
            const double lhs =i_alpha * (static_cast<double>(j) + alpha);

            const bool in_AC = (lhs <= RHS);

            const bool is_edge =(A_sub[local_pos] != 0);

            // --------------------------------------------
            // Inside A_C
            // --------------------------------------------

            if (in_AC)
            {
                Area_AC += weight;
                if (is_edge)
                    Area_EC += weight;
                else
                    Area_Ec_comp += weight;
            }

            // --------------------------------------------
            // Outside A_C
            // --------------------------------------------

            else
            {
                Area_AC_comp += weight;

                if (is_edge)
                    Area_AC_comp2 += weight;
                else
                    K2_sum += weight;
            }
        }
    }

    // ============================================================
    // 5. Density inside A_C
    // ============================================================

    const double d_c =(Area_AC > 0) ? static_cast<double>(Area_EC) / static_cast<double>(Area_AC) : 0.0;

    // ============================================================
    // 6. Fixed global outside density
    // ============================================================

    const double d_o = d_o_global;

    // ============================================================
    // 7. Local objective with fixed outside density
    // ============================================================

    constexpr double eps = 1e-12;

    const double log_like =static_cast<double>(Area_EC) *std::log(d_c + eps)+
        static_cast<double>(Area_Ec_comp) *std::log(1.0 - d_c + eps)+
        static_cast<double>(Area_AC_comp2) *std::log(d_o + eps)+
        static_cast<double>(K2_sum) *std::log(1.0 - d_o + eps);

    // ============================================================
    // 8. Return result
    // ============================================================

    result.log_like = log_like;
    result.d_c = d_c;
    result.Area_EC = Area_EC;
    result.Area_Ec_block_comp = Area_Ec_comp;
    result.valid = true;

    return result;
}

EXPORT int compute_ac_grid_global_do_sparse(
    const uint8_t* A_sub,
    int64_t nc,
    int min_gamma,
    int max_gamma,
    int min_H,
    int max_H,
    const int64_t* global_idx,
    const uint8_t* M,
    int64_t N_global,
    double d_o_global,
    int* best_gamma,
    int* best_H,
    double* best_log_like,
    double* best_d_c,
    int64_t* best_Area_EC,
    int64_t* best_Area_Ec_block_comp)
{
/*     if (!A_sub ||
        !global_idx ||
        !M ||
        !best_gamma ||
        !best_H ||
        !best_log_like ||
        !best_d_c ||
        !best_Area_EC ||
        !best_Area_Ec_block_comp ||
        nc <= 0 ||
        N_global <= 0)
    {
        return -1;
    } */

    if (d_o_global < 0.0 || d_o_global > 1.0)
        return -2;

    double global_best_ll = -std::numeric_limits<double>::infinity();
    int global_best_gamma = 1;
    int global_best_H = 0;
    double global_best_dc = 0.0;
    int64_t global_best_Area_EC = 0;
    int64_t global_best_Area_Ec_comp = 0;

    #pragma omp parallel
    {
        double local_best_ll = -std::numeric_limits<double>::infinity();
        int local_best_gamma = 1;
        int local_best_H = 0;
        double local_best_dc = 0.0;
        int64_t local_best_Area_EC = 0;
        int64_t local_best_Area_Ec_comp = 0;

        // Parallelize only the outer loop for better MSVC portability.
        #pragma omp for schedule(dynamic, 4)
        for (int gamma_test = min_gamma;gamma_test < max_gamma;  ++gamma_test)
        {
            for (int H_test = min_H;  H_test < max_H;  ++H_test)
            {
                // Same rule as the dense reference: gamma > H.
                if (gamma_test <= H_test)
                    continue;

                const double P1_check =static_cast<double>(nc - 1) + H_test- 2.0 * gamma_test;

                const double denom = static_cast<double>(nc - 1) + H_test;

                if (std::abs(denom) < 1e-12)
                    continue;

                const double P2_check =gamma_test- (2.0* static_cast<double>(nc - 1)* H_test/ denom);

                if (P1_check <= 0.0 || P2_check < 0.0)
                    continue;

                ACGridResultGlobalDO r = evaluate_one_ac_global_do(A_sub, nc, static_cast<double>(gamma_test), static_cast<double>(H_test), global_idx, M, N_global, 
                d_o_global);

                if (!r.valid)
                    continue;

                // Same >= update behavior as the dense reference.
                if (r.log_like >= local_best_ll)
                {
                    local_best_ll = r.log_like;
                    local_best_gamma = gamma_test;
                    local_best_H = H_test;
                    local_best_dc = r.d_c;
                    local_best_Area_EC = r.Area_EC;
                    local_best_Area_Ec_comp = r.Area_Ec_block_comp;
                }
            }
        }

        #pragma omp critical
        {
            if (local_best_ll >= global_best_ll)
            {
                global_best_ll = local_best_ll;
                global_best_gamma = local_best_gamma;
                global_best_H = local_best_H;
                global_best_dc = local_best_dc;
                global_best_Area_EC = local_best_Area_EC;
                global_best_Area_Ec_comp = local_best_Area_Ec_comp;
            }
        }
    }

    *best_gamma = global_best_gamma;
    *best_H = global_best_H;
    *best_log_like = global_best_ll;
    *best_d_c = global_best_dc;
    *best_Area_EC = global_best_Area_EC;
    *best_Area_Ec_block_comp = global_best_Area_Ec_comp;

    return 0;
}
