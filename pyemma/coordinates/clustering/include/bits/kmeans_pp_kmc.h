//
// Created by marscher on 11/7/17.
//

#ifndef PYEMMA_KMEANS_PP_KMC_H
#define PYEMMA_KMEANS_PP_KMC_H

#include <kmeans.h>

template<typename dtype>
typename KMeans<dtype>::np_array KMeans<dtype>::
KMeans::initCentersKMC(const KMeans::np_array &np_data, unsigned int seed,
                       unsigned int chain_len, bool afkmc2,
                       const KMeans::np_array &np_weights) const {
    auto dim = np_data.shape(1);
    std::vector<std::size_t> shape = {k, dim};
    np_array centers(shape);

    if(np_weights.is_none()) {
        np_weights = py::array_t<int>();
        // TODO: assign ones...
    }


    std::vector<std::size_t> indices(np_data.shape(0));
    for (std::size_t i = 0; i < np_data.shape(0); ++i) {
        indices.push_back(i);
    }

    std::discrete_distribution<std::size_t> dist(std::begin(np_weights), std::end(np_weights));
    std::mt19937 gen(seed);

    auto first_row_index = dist(gen);
    auto rel_row = np_data.data(first_row_index);
    // assign first center
    centers[0] = rel_row;

    std::size_t di = 0;
    np_array q;
    if (afkmc2) {
        auto min = std::numeric_limits<dtype>::max();
        //std::vector<dtype> dists;
        for (std::size_t i = 0; i < np_data.shape(0); ++i) {
            auto value = parent_t::metric->compute(&np_data(i, 0), &centers(0, 0));
            if (value <= min) {
                min = value;
                di = i;
            }
        }

    } else {
        q = std::copy(np_weights);
    }
    // norm q
    auto  q_sum = 0;
    for (int k = 0; k < q.shape(0); ++k) {
        q_sum += q[k];
    }
    q /= q_sum;

    std::vector<std::size_t> cand_ind(chain_len);
    double cand_prob, curr_prob;
    double* q_cand, p_cand, rand_a;

    for (int i = 0; i < k-1; ++i) {
        // Draw the candidate indices
        for(int j =0 ; j< chain_len; ++j) {
            cand_ind[j] = static_cast<std::size_t >(dist(gen));
        }
        // Extract the proposal probabilities
        auto q_cand = q[cand_ind];
        // compute pairwise dists
        auto dist;
        // compute potentials
        auto p_cand;

        // compute acceptance probabilities
        auto rand_a;

        for (int j = 0; j < q_cand.shape(0); ++j) {
            auto cand_prob = p_cand[j] / q_cand[j];
            if(j==0 || curr_prob == 0.0 || cand_prob / curr_prob > rand_a[j]) {
                // init new chain; Metropolis-Hastings-step
                curr_ind = j;
                curr_prob = cand_prob;
            }
        }
        rel_row = np_data[cand_ind[curr_ind], :];
        centers[i+1, :] = rel_row;
    }

    return centers;
}

#endif //PYEMMA_KMEANS_PP_KMC_H
