//
// Created by marscher on 11/7/17.
//

#ifndef PYEMMA_KMEANS_PP_KMC_H
#define PYEMMA_KMEANS_PP_KMC_H

#include <kmeans.h>
#include <functional>
#include <algorithm>

template<typename dtype>
typename KMeans<dtype>::np_array
KMeans<dtype>::initCentersKMC(const np_array &np_data, unsigned int random_seed, unsigned int chain_len,
                              bool afkmc2, np_array &np_weights,  py::object& callback) const {

    auto dim = np_data.shape(1);
    std::vector<std::size_t> shape = {k, dim};
    np_array np_centers(shape);

    if (np_weights.is_none()) {
        np_weights = py::array_t<int>(np_data.shape(0));
        auto buff = np_weights.mutable_unchecked();
        for (std::size_t i = 0; i < np_weights.shape(0); ++i) {
            buff(i) = i;
        }
    }

    auto weights = np_weights.template unchecked<1>();

    std::vector<std::size_t> indices(np_data.shape(0));
    for (std::size_t i = 0; i < np_data.shape(0); ++i) {
        indices.push_back(i);
    }

    // TODO: check distribution
    std::discrete_distribution<std::size_t> dist(&weights(0), &weights(np_weights.shape(0)));
    std::mt19937 gen(random_seed);

    auto first_row_index = dist(gen);
    auto rel_row = np_data.data(first_row_index);
    // assign first center
    np_centers[0] = rel_row;

    np_array di(dim);
    auto data = np_data.unchecked();
    auto centers = np_centers.unchecked();
    np_array q(dim);
    if (afkmc2) {
        auto min = std::numeric_limits<dtype>::max();
        //std::vector<dtype> dists;
        std::size_t min_ind = 0;
        for (std::size_t i = 0; i < np_data.shape(0); ++i) {
            auto value = parent_t::metric->compute(&data(i, 0), &centers(0, 0));
            if (value <= min) {
                min = value;
                min_ind = i;
            }
        }
        auto diData = di.mutable_data();
        // assign min center to di
        std::copy(&centers(min_ind, 0), &centers(min_ind, dim), diData);
        // di * w
        std::transform(diData, diData + dim, weights.data(0), diData, [](const auto d, const auto w) {
            return d * w;
        });

    } else {
        std::copy(weights.data(0), weights.data(0) + dim, q.mutable_data());
    }
    // norm q
    auto q_sum = static_cast<dtype>(0);
    for (int k = 0; k < q.shape(0); ++k) {
        q_sum += q.at(k);
    }
    std::transform(q.data(), q.data() + dim, q.mutable_data(), [q_sum](auto qq) {
        return qq / q_sum;
    });

    std::vector<std::size_t> cand_ind(chain_len);
    double cand_prob, curr_prob;
    std::size_t curr_ind = 0;

    // acceptance distribution
    std::uniform_real_distribution<dtype> acceptance_dist;
    std::mt19937 acceptance_gen(random_seed);

    for (int i = 0; i < k - 1; ++i) {
        // Draw the candidate indices
        for (int j = 0; j < chain_len; ++j) {
            cand_ind[j] = static_cast<std::size_t >(dist(gen));
        }
        // Extract the proposal probabilities
        // TODO: fancy indexing possible?
        auto q_cand = q.at(cand_ind);
        // compute pairwise distances for each candidate
        for (const auto candidateIndex : cand_ind) {
            auto min = std::numeric_limits<dtype>::max();
            for (auto j = 0U; j < i + 1; ++j) {
                auto value = parent_t::metric->compute(&data(candidateIndex, 0), &centers(j, 0));
                if (value < min) {
                    min = value;
                }
            }
        }
        // compute potentials

        // di * w
        std::transform(diData, diData + dim, weights.data(0), diData, [](const auto d, const auto w) {
            return d * w;
        });

        auto p_cand;

        // compute acceptance probabilities
        auto rand_a;
        for (int j = 0; j < chain_len; ++j) {
            rand_a[j] = acceptance_dist(acceptance_gen);
        }

        for (int j = 0; j < q_cand.shape(0); ++j) {
            auto cand_prob = p_cand[j] / q_cand[j];
            if (j == 0 || curr_prob == 0.0 || cand_prob / curr_prob > rand_a[j]) {
                // init new chain; Metropolis-Hastings-step
                curr_ind = j;
                curr_prob = cand_prob;
            }
        }
        //rel_row = np_data[cand_ind[curr_ind], :];
        //np_centers[i+1, :] = rel_row;
        if (! callback.is_none()) {
            callback();
        }
    }

    return np_centers;

}

#endif //PYEMMA_KMEANS_PP_KMC_H
