//
// Created by marscher on 4/3/17.
//

#ifndef PYEMMA_KMEANS_H
#define PYEMMA_KMEANS_H

#include <utility>

#include "Clustering.h"

namespace py = pybind11;


template<typename dtype>
class KMeans : public ClusteringBase<dtype> {
public:
    using parent_t = ClusteringBase<dtype>;
    using np_array = py::array_t<dtype, py::array::c_style | py::array::forcecast>;

    KMeans(unsigned int k,
           const std::string &metric,
           size_t input_dimension,
           py::function callback) : ClusteringBase<dtype>(metric, input_dimension), k(k),
                                    callback(std::move(callback)) {}

    /**
     * performs kmeans clustering on the given data chunk, provided a list of centers.
     * @param np_chunk
     * @param np_centers
     * @param n_threads
     * @return updated centers.
     */
    np_array cluster(const np_array & /*np_chunk*/, const np_array & /*np_centers*/, int /*n_threads*/) const;

    /**
     * evaluate the quality of the centers
     *
     * @return
     */
    dtype costFunction(const np_array & /*np_data*/, const np_array & /*np_centers*/, int /*n_threads*/) const;

    /**
     * kmeans++ initialisation
     * @param np_data
     * @param random_seed
     * @param n_threads
     * @return init centers.
     */
    np_array initCentersKMpp(const np_array& /*np_data*/, unsigned int /*random_seed*/, int /*n_threads*/) const;


    /**
     * kmeans++ initialisation
     * @param np_data
     * @param random_seed
     * @param chain_len
     * @param afkmc2
     * @param weights
     * @return init centers.
     */
    np_array initCentersKMC(const np_array& /*np_data*/, unsigned int /*random_seed*/, unsigned int /*chain len */,
                            bool afkmc2, const np_array&np_weights) const;

    /**
     * call back function to inform about progress
     * @param callback None or Python function.
     */
    void set_callback(const py::function &callback) { this->callback = callback; }

protected:
    unsigned int k;
    py::function callback;

};

#include "bits/kmeans_bits.h"
#include "bits/kmeans_pp_kmc.h"

#endif //PYEMMA_KMEANS_H
