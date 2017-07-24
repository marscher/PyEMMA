//
// Created by marscher on 7/24/17.
//


#ifndef PYEMMA_KMEANS_BITS_H_H
#define PYEMMA_KMEANS_BITS_H_H

#include "kmeans.h"
#include <pybind11/pytypes.h>

template<typename dtype>
py::array_t <dtype> KMeans<dtype>::cluster(const py::array_t <dtype, py::array::c_style> &np_chunk,
                                           const py::array_t <dtype, py::array::c_style> &np_centers) {
    int debug;

    size_t i, j;
    debug = 0;

    if (np_chunk.ndim() != 2) { throw std::runtime_error("Number of dimensions of \"chunk\" isn\'t 2."); }

    size_t N_frames = np_chunk.shape(0);
    size_t dim = np_chunk.shape(1);

    if (dim == 0) {
        throw std::invalid_argument("chunk dimension must be larger than zero.");
    }

    auto chunk = np_chunk.template unchecked<2>();
    if (debug) printf("done with N_frames=%zd, dim=%zd\n", N_frames, dim);

    /* import list of cluster centers */
    if (debug) printf("KMEANS: importing list of cluster centers...");
    size_t N_centers = np_centers.shape(0);

    auto centers = np_centers.template unchecked<2>();

    if (debug) printf("done, k=%zd\n", N_centers);
    /* initialize centers_counter and new_centers with zeros */
    std::vector<int> centers_counter(N_centers, 0);
    std::vector<dtype> new_centers(N_centers * dim, 0.0);

    /* do the clustering */
    if (debug) printf("KMEANS: performing the clustering...");
    int *centers_counter_p = centers_counter.data();
    dtype *new_centers_p = new_centers.data();
    dtype mindist;
    size_t closest_center_index = 0;
    dtype d;
    for (i = 0; i < N_frames; i++) {
        mindist = std::numeric_limits<dtype>::max();
        for (j = 0; j < N_centers; ++j) {
            d = parent_t::metric->compute(&chunk(i, 0), &centers(j, 0));
            if (d < mindist) {
                mindist = d;
                closest_center_index = j;
            }
        }
        (*(centers_counter_p + closest_center_index))++;
        for (j = 0; j < dim; j++) {
            new_centers[closest_center_index * dim + j] += chunk(i, j);
        }
    }

    for (i = 0; i < N_centers; i++) {
        if (*(centers_counter_p + i) == 0) {
            for (j = 0; j < dim; j++) {
                (*(new_centers_p + i * dim + j)) = centers(i, j);
            }
        } else {
            for (j = 0; j < dim; j++) {
                (*(new_centers_p + i * dim + j)) /= (*(centers_counter_p + i));
            }
        }
    }
    if (debug) printf("done\n");

    if (debug) printf("KMEANS: creating return_new_centers...");
    std::vector<size_t> shape = {N_centers, dim};
    py::array_t <dtype> return_new_centers(shape);
    void *arr_data = return_new_centers.mutable_data();
    if (debug) printf("done\n");
    // TODO: this is not needed anymore, because we could modify the centers in place?
    /* Need to copy the data of the malloced buffer to the PyObject
       since the malloced buffer will disappear after the C extension is called. */
    if (debug) printf("KMEANS: attempting memcopy...");
    memcpy(arr_data, new_centers_p, return_new_centers.itemsize() * N_centers * dim);
    if (debug) printf("done\n");
    return return_new_centers;
}

template<typename dtype>
dtype KMeans<dtype>::costFunction(const np_array& np_data, const np_array& np_centers) {
    std::size_t n_frames;
    auto data = np_data.template unchecked<2>();
    auto centers = np_centers.template unchecked<2>();

    dtype value = 0.0;
    n_frames = np_data.shape(0);

    for (size_t r = 0; r < np_centers.shape(0); r++) {
        for (size_t i = 0; i < n_frames; i++) {
            value += parent_t::metric->compute(&data(i, 0), &centers(r, 0));
        }
    }
    return value;
}


template<typename dtype>
typename KMeans<dtype>::np_array KMeans<dtype>::
initCentersKMpp(const KMeans::np_array& np_data, unsigned int random_seed) {
    size_t centers_found = 0, first_center_index, n_trials;
    int some_not_done;
    dtype d;
    dtype dist_sum = 0.0;
    dtype sum;
    size_t dim, n_frames;
    size_t i, j;
    int *taken_points = nullptr;
    int best_candidate = -1;
    dtype best_potential = std::numeric_limits<dtype>::max();
    std::vector<int> next_center_candidates;
    std::vector<dtype> next_center_candidates_rand;
    std::vector<dtype> next_center_candidates_potential;
    std::vector<dtype> init_centers, squared_distances;
    std::vector<dtype> arr_data;

    /* set random seed */
    //printf("initkmpp: set seed to %u\n", random_seed);
    srand(random_seed);

    n_frames = np_data.shape(0);
    dim = np_data.shape(1);
    auto data = np_data.template unchecked<2>();
    /* number of trials before choosing the data point with the best potential */
    n_trials = 2 + (int) log(k);

    /* allocate space for the index giving away which point has already been used as a cluster center */
    if (!(taken_points = (int *) calloc(n_frames, sizeof(int)))) {
        throw std::bad_alloc();
    }
    /* allocate space for the array holding the cluster centers to be returned */
    init_centers.resize(k * dim);
    /* allocate space for the array holding the squared distances to the assigned cluster centers */
    squared_distances.resize(n_frames);

    /* candidates allocations */
    next_center_candidates.resize(n_trials);
    next_center_candidates_rand.resize(n_trials);
    next_center_candidates_potential.resize(n_trials);

    /* pick first center randomly */
    first_center_index = rand() % n_frames;
    /* and mark it as assigned */
    taken_points[first_center_index] = 1;
    /* write its coordinates into the init_centers array */
    for (j = 0; j < dim; j++) {
        (*(init_centers.data() + centers_found * dim + j)) = data(first_center_index, j);
    }
    /* increase number of found centers */
    centers_found++;
    /* perform callback */
    if (not py::isinstance<py::none>(callback)) {
        callback();
    }

    /* iterate over all data points j, measuring the squared distance between j and the initial center i: */
    /* squared_distances[i] = distance(x_j, x_i)*distance(x_j, x_i) */
    for (i = 0; i < n_frames; i++) {
        if (i != first_center_index) {
            auto value = parent_t::metric->compute(&data(i, dim), &data(first_center_index, 0));
            squared_distances[i] = value * value;
            /* build up dist_sum which keeps the sum of all squared distances */
            dist_sum += d;
        }
    }

    /* keep picking centers while we do not have enough of them... */
    while (centers_found < k) {

        /* initialize the trials random values by the D^2-weighted distribution */
        for (j = 0; j < n_trials; j++) {
            next_center_candidates[j] = -1;
            next_center_candidates_rand[j] = dist_sum * ((dtype) rand() / (dtype) RAND_MAX);
            next_center_candidates_potential[j] = 0.0;
        }

        /* pick candidate data points corresponding to their random value */
        sum = 0.0;
        for (i = 0; i < n_frames; i++) {
            if (!taken_points[i]) {
                sum += squared_distances[i];
                some_not_done = 0;
                for (j = 0; j < n_trials; j++) {
                    if (next_center_candidates[j] == -1) {
                        if (sum >= next_center_candidates_rand[j]) {
                            next_center_candidates[j] = i;
                        } else {
                            some_not_done = 1;
                        }
                    }
                }
                if (!some_not_done) break;
            }
        }

        /* now find the maximum squared distance for each trial... */
        for (i = 0; i < n_frames; i++) {
            if (!taken_points[i]) {
                for (j = 0; j < n_trials; j++) {
                    if (next_center_candidates[j] == -1) break;
                    if (next_center_candidates[j] != i) {
                        auto value = parent_t::metric->compute(&data(i, 0), &data(next_center_candidates[j], 0));
                        d = value * value;
                        if (d < squared_distances[i]) {
                            next_center_candidates_potential[j] += d;
                        } else {
                            next_center_candidates_potential[j] += squared_distances[i];
                        }
                    }
                }
            }
        }

        /* ... and select the best candidate by the minimum value of the maximum squared distances */
        best_candidate = -1;
        best_potential = std::numeric_limits<dtype>::max();
        for (j = 0; j < n_trials; j++) {
            if (next_center_candidates[j] != -1 && next_center_candidates_potential[j] < best_potential) {
                best_potential = next_center_candidates_potential[j];
                best_candidate = next_center_candidates[j];
            }
        }

        /* if for some reason we did not find a best candidate, just take the next available point */
        if (best_candidate == -1) {
            for (i = 0; i < n_frames; i++) {
                if (!taken_points[i]) {
                    best_candidate = i;
                    break;
                }
            }
        }

        /* check if best_candidate was set, otherwise break to avoid an infinite loop should things go wrong */
        if (best_candidate >= 0) {
            /* write the best_candidate's components into the init_centers array */
            for (j = 0; j < dim; j++) {
                (*(init_centers.data() + centers_found * dim + j)) = data(best_candidate, j);
            }
            /* increase centers_found */
            centers_found++;
            /* perform the callback */
            if (not py::isinstance<py::none>(callback)) {
                callback();
            }
            /* mark the data point as assigned center */
            taken_points[best_candidate] = 1;
            /* update the sum of squared distances by removing the assigned center */
            dist_sum -= squared_distances[best_candidate];

            /* if we still have centers to assign, the squared distances array has to be updated */
            if (centers_found < k) {
                /* Check for each data point if its squared distance to the freshly added center is smaller than */
                /* the squared distance to the previously picked centers. If so, update the squared_distances */
                /* array by the new value and also update the dist_sum value by removing the old value and adding */
                /* the new one. */
                for (i = 0; i < n_frames; i++) {
                    if (!taken_points[i]) {
                        auto value = parent_t::metric->compute(&data(i, 0), &data(best_candidate, 0));
                        d = value * value;
                        if (d < squared_distances[i]) {
                            dist_sum += d - squared_distances[i];
                            squared_distances[i] = d;
                        }
                    }
                }
            }
        } else {
            break;
        }
    }

    /* create the output objects */
    std::vector<size_t> shape = {k, dim};
    py::array_t <dtype, py::array::c_style> ret_init_centers(shape);

    memcpy(ret_init_centers.mutable_data(), arr_data.data(), arr_data.size());
    return ret_init_centers;
}

#endif //PYEMMA_KMEANS_BITS_H_H