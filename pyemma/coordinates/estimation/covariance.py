# This file is part of PyEMMA.
#
# Copyright (c) 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import

import numpy as np
import numbers
from math import log
import random

from pyemma.util.annotators import deprecated
from pyemma.util.types import is_float_vector, ensure_float_vector
from pyemma.coordinates.data._base.streaming_estimator import StreamingEstimator
from pyemma._base.progress import ProgressReporter
from pyemma._ext.variational.estimators.running_moments import running_covar
from pyemma.coordinates.data._base.streaming_estimator import StreamingEstimator
from pyemma.util.metrics import vamp_score
from pyemma.util.types import is_float_vector, ensure_float_vector

__all__ = ['LaggedCovariance']

__author__ = 'paul, nueske, marscher, clonker'


class LaggedCovariance(StreamingEstimator, ProgressReporter):
    r"""Compute lagged covariances between time series.

     Parameters
     ----------
     c00 : bool, optional, default=True
         compute instantaneous correlations over the first part of the data. If lag==0, use all of the data.
         Makes the C00_ attribute available.
     c0t : bool, optional, default=False
         compute lagged correlations. Does not work with lag==0.
         Makes the C0t_ attribute available.
     ctt : bool, optional, default=False
         compute instantaneous correlations over the time-shifted chunks of the data. Does not work with lag==0.
         Makes the Ctt_ attribute available.
     remove_constant_mean : ndarray(N,), optional, default=None
         substract a constant vector of mean values from time series.
     remove_data_mean : bool, optional, default=False
         substract the sample mean from the time series (mean-free correlations).
     reversible : bool, optional, default=False
         symmetrize correlations.
     bessel : bool, optional, default=True
         use Bessel's correction for correlations in order to use an unbiased estimator
     sparse_mode : str, optional, default='auto'
         one of:
             * 'dense' : always use dense mode
             * 'auto' : automatic
             * 'sparse' : always use sparse mode if possible
     modify_data : bool, optional, default=False
         If remove_data_mean=True, the mean will be removed in the input data, without creating an independent copy.
         This option is faster but should only be selected if the input data is not used elsewhere.
     lag : int, optional, default=0
         lag time. Does not work with c0t=True or ctt=True.
     weights : trajectory weights.
         one of:
             * None :    all frames have weight one.
             * float :   all frames have the same specified weight.
             * object:   an object that possesses a .weight(X) function in order to assign weights to every
                         time step in a trajectory X.
             * list of arrays: ....

     stride: int, optional, default = 1
         Use only every stride-th time step. By default, every time step is used.
     skip : int, optional, default=0
         skip the first initial n frames per trajectory.
     chunksize : int, optional, default=None
         The chunk size at which the input files are being processed.

     """
    def __init__(self, c00=True, c0t=False, ctt=False, remove_constant_mean=None, remove_data_mean=False, reversible=False,
                 bessel=True, sparse_mode='auto', modify_data=False, lag=0, weights=None, stride=1, skip=0,
                 chunksize=None, ncov_max=float('inf')):
        super(LaggedCovariance, self).__init__(chunksize=chunksize)

        if (c0t or ctt) and lag == 0:
            raise ValueError("lag must be positive if c0t=True or ctt=True")

        if remove_constant_mean is not None and remove_data_mean:
            raise ValueError('Subtracting the data mean and a constant vector simultaneously is not supported.')
        if remove_constant_mean is not None:
            remove_constant_mean = ensure_float_vector(remove_constant_mean)
        self.set_params(c00=c00, c0t=c0t, ctt=ctt, remove_constant_mean=remove_constant_mean,
                        remove_data_mean=remove_data_mean, reversible=reversible,
                        sparse_mode=sparse_mode, modify_data=modify_data, lag=lag,
                        bessel=bessel,
                        weights=weights, stride=stride, skip=skip, ncov_max=ncov_max)

        self._rc = None
        self._used_data = 0

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        from pyemma.coordinates.data import DataInMemory
        import types

        if is_float_vector(value):
            value = DataInMemory(value)
        elif isinstance(value, (list, tuple)):
            value = DataInMemory(value)
        elif isinstance(value, numbers.Integral):
            value = float(value) if value is not None else 1.0
        elif hasattr(value, 'weights') and type(getattr(value, 'weights')) == types.MethodType:
            from pyemma.coordinates.data._base.transformer import StreamingTransformer
            class compute_weights_streamer(StreamingTransformer):
                def __init__(self, func):
                    super(compute_weights_streamer, self).__init__()
                    self.func = func
                def dimension(self):
                    return 1
                def _transform_array(self, X):
                    return self.func.weights(X)
                def describe(self): pass

            value = compute_weights_streamer(value)

        self._weights = value

    def _init_covar(self, partial_fit, n_chunks):
        nsave = min(int(max(log(n_chunks, 2), 2)), self.ncov_max)
        if self._rc is not None and partial_fit:
            # check storage size vs. n_chunks of the new iterator
            old_nsave = self.nsave
            if old_nsave < nsave:
                self.logger.info("adapting storage size")
                self.nsave = nsave
        else:
            # in case we do a one shot estimation, we want to re-initialize running_covar
            self._logger.debug("using %s moments for %i chunks", nsave, n_chunks)
            self._rc = running_covar(xx=self.c00, xy=self.c0t, yy=self.ctt,
                                     remove_mean=self.remove_data_mean, symmetrize=self.reversible,
                                     sparse_mode=self.sparse_mode, modify_data=self.modify_data, nsave=nsave)

    def _estimate(self, iterable, partial_fit=False):
        indim = iterable.dimension()
        if not indim:
            raise ValueError("zero dimension from data source!")

        if not any(iterable.trajectory_lengths(stride=self.stride, skip=self.lag+self.skip) > 0):
            if partial_fit:
                self.logger.warn("Could not use data passed to partial_fit(), "
                                 "because no single data set [longest=%i] is longer than lag+skip [%i]",
                                 max(iterable.trajectory_lengths(self.stride, skip=self.skip)), self.lag+self.skip)
                return self
            else:
                raise ValueError("None single dataset [longest=%i] is longer than"
                                 " lag+skip [%i]." % (max(iterable.trajectory_lengths(self.stride, skip=self.skip)),
                                                      self.lag+self.skip))

        self.logger.debug("will use %s total frames for %s",
                          iterable.trajectory_lengths(self.stride, skip=self.skip), self.name)

        it = iterable.iterator(lag=self.lag, return_trajindex=False, stride=self.stride, skip=self.skip,
                               chunk=self.chunksize if not partial_fit else 0)
        # iterator over input weights
        if hasattr(self.weights, 'iterator'):
            if hasattr(self.weights, '_transform_array'):
                self.weights.data_producer = iterable
            it_weights = self.weights.iterator(lag=0, return_trajindex=False, stride=self.stride, skip=self.skip,
                                               chunk=self.chunksize if not partial_fit else 0)
            if it_weights.number_of_trajectories() != iterable.number_of_trajectories():
                raise ValueError("number of weight arrays did not match number of input data sets. {} vs. {}"
                                 .format(it_weights.number_of_trajectories(), iterable.number_of_trajectories()))
        else:
            # if we only have a scalar, repeat it.
            import itertools
            it_weights = itertools.repeat(self.weights)

        # TODO: we could possibly optimize the case lag>0 and c0t=False using skip.
        # Access how much iterator hassle this would be.
        #self.skipped=0
        with it:
            self._progress_register(it.n_chunks, "calculate covariances", 0)
            self._init_covar(partial_fit, it.n_chunks)
            for data, weight in zip(it, it_weights):
                if self.lag != 0:
                    X, Y = data
                else:
                    X, Y = data, None

                if weight is not None:
                    if isinstance(weight, np.ndarray):
                        weight = weight.squeeze()[:len(X)]
                        # TODO: if the weight is exactly zero it makes not sense to add the chunk to running moments.
                        # however doing so, leads to wrong results...
                        # if np.all(np.abs(weight) < np.finfo(np.float).eps):
                        #     #print("skip")
                        #     self.skipped += len(X)
                        #     continue
                if self.remove_constant_mean is not None:
                    X = X - self.remove_constant_mean[np.newaxis, :]
                    if Y is not None:
                        Y = Y - self.remove_constant_mean[np.newaxis, :]

                try:
                    self._rc.add(X, Y, weights=weight)
                except MemoryError:
                    raise MemoryError('Covariance matrix does not fit into memory. '
                                      'Input is too high-dimensional ({} dimensions). '.format(X.shape[1]))
                self._progress_update(1, stage=0)

        if partial_fit:
            self._used_data += len(it)

    def partial_fit(self, X):
        """ incrementally update the estimates

        Parameters
        ----------
        X: array, list of arrays, PyEMMA reader
            input data.
        """
        from pyemma.coordinates import source

        self._estimate(source(X), partial_fit=True)
        self._estimated = True

        return self

    @property
    def mean(self):
        self._check_estimated()
        return self._rc.mean_X()

    @property
    def mean_tau(self):
        self._check_estimated()
        return self._rc.mean_Y()

    @property
    @deprecated('Please use the attribute "C00_".')
    def cov(self):
        self._check_estimated()
        return self._rc.cov_XX(bessel=self.bessel)

    @property
    def C00_(self):
        """ Instantaneous covariance matrix """
        self._check_estimated()
        return self._rc.cov_XX(bessel=self.bessel)

    @property
    @deprecated('Please use the attribute "C0t_".')
    def cov_tau(self):
        self._check_estimated()
        return self._rc.cov_XY(bessel=self.bessel)

    @property
    def C0t_(self):
        """ Time-lagged covariance matrix """
        self._check_estimated()
        return self._rc.cov_XY(bessel=self.bessel)

    @property
    def Ctt_(self):
        """ Covariance matrix of the time shifted data"""
        self._check_estimated()
        return self._rc.cov_YY(bessel=self.bessel)

    @property
    def nsave(self):
        if self.c00:
            return self._rc.storage_XX.nsave
        elif self.c0t:
            return self._rc.storage_XY.nsave

    @nsave.setter
    def nsave(self, ns):
        # potential bug? set nsave between partial fits?
        if self.c00:
            if self._rc.storage_XX.nsave <= ns:
                self._rc.storage_XX.nsave = ns
        if self.c0t:
            if self._rc.storage_XY.nsave <= ns:
                self._rc.storage_XY.nsave = ns
        if self.ctt:
            if self._rc.storage_YY.nsave <= ns:
                self._rc.storage_YY.nsave = ns


class _ShuffleSplit(object):

    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def split(self, covs):
        n = len(covs)
        full = np.arange(n)
        for _ in range(self.n_splits):
            train = np.random.choice(full, int(n / 2), replace=False)
            test = np.setdiff1d(full, train)
            yield train, test


class Covariances(StreamingEstimator, ProgressReporter):
    """
    Parameters
    ----------

    tau: int, default=5000
        size of running blocks of input stream.

    mode: str, default="sliding"

    """

    def __init__(self, n_covs, n_save=5, tau=5000, shift=10, stride=1, mode='sliding', assign_to_covs='random'):
        super(Covariances, self).__init__()
        if mode not in ('sliding', 'linear'):
            raise ValueError('unsupported mode: %s' % mode)
        self.set_params(mode=mode, tau=tau, shift=shift, n_covs=n_covs, n_save=n_save,
                        assign_to_covs=assign_to_covs,
                        stride=stride)

    class _LinearCovariancesSplit(object):

        def __init__(self, block_size, shift, stride):
            self.block_size = block_size
            self.shift = shift
            self.stride = stride

        def n_chunks(self, iterable):
            return iterable.n_chunks(self.block_size, skip=self.shift, stride=self.stride) - 1

        def split(self, iterable):
            with iterable.iterator(chunk=self.block_size, return_trajindex=False, skip=self.shift,
                                   stride=self.stride) as it:
                current_chunk, next_chunk = None, None

                for current_data in it:
                    next_chunk = current_data

                    if current_chunk is not None:
                        yield current_chunk[:len(next_chunk)], next_chunk

                    if not it.last_chunk_in_traj:
                        current_chunk = next_chunk
                    else:
                        current_chunk, next_chunk = None, None

    class _SlidingCovariancesSplit(object):
        def __init__(self, block_size, offset, stride):
            self.block_size = block_size
            self.offset = offset
            self.stride = stride

        def n_chunks(self, iterable):
            n1 = iterable.n_chunks(2 * self.block_size - 1, stride=self.stride, skip=self.offset)
            n2 = iterable.n_chunks(2 * self.block_size - 1, stride=self.stride, skip=self.offset + self.block_size)
            return min(n1, n2)

        def split(self, iterable):
            with iterable.iterator(lag=self.block_size, chunk=2 * self.block_size - 1, return_trajindex=False,
                                   skip=self.offset, stride=self.stride) as it:
                for first_chunk, lagged_chunk in it:
                    yield first_chunk[:len(lagged_chunk)], lagged_chunk

    def _estimate(self, iterable):
        self.covs_ = np.array([running_covar(xx=True, xy=True, yy=True, remove_mean=False,
                                             symmetrize=False, sparse_mode='auto',
                                             modify_data=False, nsave=self.n_save) for _ in range(self.n_covs)])

        if self.mode == 'sliding':
            splitter = self._SlidingCovariancesSplit(self.tau, self.shift, self.stride)
        elif self.mode == 'linear':
            splitter = self._LinearCovariancesSplit(self.tau, self.shift, self.stride)
        else:
            raise NotImplementedError("unsupported mode: %s" % self.mode)

        self._progress_register(splitter.n_chunks(iterable), "calculate covariances", 0)

        if self.assign_to_covs == 'round_robin':
            idx = 0
            def index():
                nonlocal idx
                res = idx % len(self.covs_)
                idx += 1
                return res

        elif self.assign_to_covs == 'random':
            def index():
                return random.randint(0, len(self.covs_) - 1)
        else:
            raise NotImplementedError('unknown assign_to_covs mode: %s' %self.assign_to_covs)

        for X, Y in splitter.split(iterable):
            index_ = index()
            self.covs_[index_].add(X, Y)
            self._progress_update(1, stage=0)

        self.covs_ = np.array(list(filter(lambda c: len(c.storage_XX.storage) > 0, self.covs_)))
        self.n_covs_ = len(self.covs_)

        if self.n_covs != self.n_covs_:
            self.logger.info("truncated covariance matrices due to lack of data (%s -> %s)", self.n_covs, self.n_covs_)

        self._progress_force_finish(stage=0)

    def _aggregate(self, selection, bessel=True):
        covs = self.covs_[selection]
        if len(covs) == 0:
            raise ValueError("all the selected (%s) covariance matrices were empty!" % selection)
        old_weights_xx = [c.weight_XX() for c in covs]
        old_weights_xy = [c.weight_XY() for c in covs]
        old_weights_yy = [c.weight_YY() for c in covs]
        cumulative_weight_xx = sum(old_weights_xx)
        cumulative_weight_xy = sum(old_weights_xy)
        cumulative_weight_yy = sum(old_weights_yy)
        for c in covs:
            if len(c.storage_XX.storage) > 0:
                c.storage_XX.moments.w = cumulative_weight_xx
            if len(c.storage_XY.storage) > 0:
                c.storage_XY.moments.w = cumulative_weight_xy
            if len(c.storage_YY.storage) > 0:
                c.storage_YY.moments.w = cumulative_weight_yy
        c00 = sum(c.cov_XX(bessel=bessel) for c in covs)
        c01 = sum(c.cov_XY(bessel=bessel) for c in covs)
        c11 = sum(c.cov_YY(bessel=bessel) for c in covs)
        for idx, c in enumerate(covs):
            if len(c.storage_XX.storage) > 0:
                c.storage_XX.storage[0].w = old_weights_xx[idx]
            if len(c.storage_XY.storage) > 0:
                c.storage_XY.storage[0].w = old_weights_xy[idx]
            if len(c.storage_YY.storage) > 0:
                c.storage_YY.storage[0].w = old_weights_yy[idx]
        return c00, c01, c11

    def score(self, test_covs, train_covs, k=5, scoring_method='vamp2'):
        # split test and train test sets from input
        c00_test, c01_test, c11_test = self._aggregate(test_covs)
        c00_train, c01_train, c11_train = self._aggregate(train_covs)
        return vamp_score(K=c01_train, C00_test=c00_test, C0t_test=c01_test, Ctt_test=c11_test,
                          C00_train=c00_train, C0t_train=c01_train, Ctt_train=c11_train,
                          k=k, score=scoring_method)

    def score_cv(self, n=10, k=None, scoring_method='vamp2', splitter='shuffle'):
        self._progress_register(n, "score cv", stage="cv")
        self._progress_update(0, stage="cv")
        scores = []

        if splitter == 'shuffle':
            splitter = _ShuffleSplit(n)
        elif not (hasattr(splitter, 'split') and callable(splitter.split)):
            raise ValueError("splitter must be either \"split\" or splitter instance with split(X) method")

        for covs_train, covs_test in splitter.split(self.covs_):
            score = self.score(covs_train, covs_test, k=k, scoring_method=scoring_method)
            scores.append(score)
            self._progress_update(1, stage="cv")
        self._progress_force_finish(stage="cv")
        return np.array(scores)

    def __getstate__(self):
        res = {}
        res.update(self.get_params())
        res['covs_'] = self.covs_
        res['n_covs_'] = self.n_covs_
        return res
