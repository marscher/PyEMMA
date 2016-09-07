/**
 * minRMSD function wrappers around mdtraj, which should be built as a dynamic library
 * to avoid the mdtraj dependency during build time.
 */

#include <clustering.h>
#include <theobald_rmsd.h>
#include <center.h>

/*
 * minRMSD distance function
 * a: centers
 * b: frames
 * n: dimension of one frame
 * buffer_a: pre-allocated buffer to store a copy of centers
 * buffer_b: pre-allocated buffer to store a copy of frames
 * trace_a_precalc: pre-calculated trace to centers (pointer to one value)
 */
#ifdef __cplusplus
extern "C" {
#endif

float minRMSD_distance(float *SKP_restrict a, float *SKP_restrict b, size_t n,
                       float *SKP_restrict buffer_a, float *SKP_restrict buffer_b,
                       float *trace_a_precalc)
{
    float msd;
    float trace_a, trace_b;

    if (! trace_a_precalc) {
        memcpy(buffer_a, a, n*sizeof(float));
        memcpy(buffer_b, b, n*sizeof(float));

        inplace_center_and_trace_atom_major(buffer_a, &trace_a, 1, n/3);
        inplace_center_and_trace_atom_major(buffer_b, &trace_b, 1, n/3);

    } else {
        // only copy b, since a has been pre-centered,
        memcpy(buffer_b, b, n*sizeof(float));
        inplace_center_and_trace_atom_major(buffer_b, &trace_b, 1, n/3);
        trace_a = *trace_a_precalc;
    }

    msd = msd_atom_major(n/3, n/3, a, buffer_b, trace_a, trace_b, 0, NULL);
    return sqrt(msd);
}

void inplace_center_and_trace_atom_major_cluster_centers_impl(float* centers_precentered, float* traces_centers_p,
    const int N_centers, const int dim) {
    int j;
    for (j = 0; j < N_centers; ++j) {
        inplace_center_and_trace_atom_major(&centers_precentered[j*dim],
                                            &traces_centers_p[j], 1, dim/3);
    }
}

#ifdef __cplusplus
}
#endif