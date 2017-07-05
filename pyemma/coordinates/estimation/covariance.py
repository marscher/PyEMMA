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

import numbers
from itertools import count
from math import log

import numpy as np
from scipy.sparse.linalg import svds

from pyemma._base.progress import ProgressReporter
from pyemma._ext.variational.estimators.running_moments import running_covar
from pyemma.coordinates.data._base.streaming_estimator import StreamingEstimator
from pyemma.util.metrics import vamp_score
from pyemma.util.types import is_float_vector, ensure_float_vector
from pyemma.util.linalg import mdot

__all__ = ['LaggedCovariance']

__author__ = 'paul, nueske, marscher, clonker'


class LaggedCovariance(StreamingEstimator, ProgressReporter):
    r"""Compute lagged covariances between time series.

     Parameters
     ----------
     c00 : bool, optional, default=True
         compute instantaneous correlations over the first part of the data. If lag==0, use all of the data.
     c0t : bool, optional, default=False
         compute lagged correlations. Does not work with lag==0.
     ctt : bool, optional, default=False
         compute instantaneous correlations over the second part of the data. Does not work with lag==0.
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
        else: # in case we do a one shot estimation, we want to re-initialize running_covar
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
    def cov(self):
        self._check_estimated()
        return self._rc.cov_XX(bessel=self.bessel)

    @property
    def cov_tau(self):
        self._check_estimated()
        return self._rc.cov_XY(bessel=self.bessel)

    @property
    def cov_tau_tau(self):
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


class LinearCovariancesSplit(object):

    def __init__(self, block_size):
        self.block_size = block_size

    def split(self, X):
        with X.iterator(chunk=self.block_size, return_trajindex=True) as it:
            X_, Y_ = None, None

            for itraj, X in it:
                Y_ = X

                if X_ is not None and len(X_) == self.block_size and len(Y_) == self.block_size:
                    yield itraj, X_, Y_

                if not it.last_chunk_in_traj:
                    X_ = Y_
                else:
                    X_, Y_ = None, None


class SlidingCovariancesSplit(object):
    def __init__(self, block_size):
        self.block_size = block_size

    def split(self, iterable):
        with iterable.iterator(lag=self.block_size, chunk=2 * self.block_size - 1, return_trajindex=True) as it:
            for itraj, X, Y in it:
                # only return a result, if block_size matches.
                if len(Y) == 2 * self.block_size - 1 and len(X) == 2 * self.block_size - 1:
                    yield itraj, X, Y


class DecomposedCovPair(object):
    _ids = count(0)

    def __init__(self, U, S, V, UU, SS, VV):
        self._V = V
        self._VV = VV
        self._S = S
        self._SS = SS
        self._U_Uprime = np.matmul(U.T, UU)
        self._N = 1 #len(U) -1
        self.id = next(DecomposedCovPair._ids)

    def __repr__(self):
        return '[DecomposedCovPair <{id}>]'.format(id=self.id)

    __str__ = __repr__

    @property
    def C00(self):
        res = mdot(self._V, np.diag(self._S**2), self._V.T) / self._N
        return res

    @property
    def C01(self):
        return mdot(self._V, np.diag(self._S), self._U_Uprime, np.diag(self._SS), self._VV.T) / self._N

    @property
    def C11(self):
        S = np.diag(self._SS**2)
        return mdot(self._VV, S, self._VV.T) / self._N

    @property
    def nbytes(self):
        return self._S.nbytes + self._SS.nbytes + self._U_Uprime.nbytes + self._V.nbytes + self._VV.nbytes

    def combine(self, others):
        C00_new = self.C00.copy()
        C01_new = self.C01.copy()
        C11_new = self.C11.copy()
        for c in others:
            C00_new += c.C00
            C01_new += c.C01
            C11_new += c.C11

        C00_new /= len(others)
        C01_new /= len(others)
        C11_new /= len(others)

        return C00_new, C01_new, C11_new


class Covariances(StreamingEstimator):
    """
    Parameters
    ----------

    k : int, default=6
        rank of sparse svd of input blocks

    block_size: int, default=5000
        size of running blocks of input stream.

    mode: str, default="sliding"


    """

    def __init__(self, k=6, block_size=5000, mode='sliding'):
        super(Covariances, self).__init__()
        if mode not in ('sliding', 'linear'):
            raise ValueError('unsupported mode: %s' % mode)
        self.set_params(k=k, mode=mode, block_size=block_size)

    def _process(self, itraj, X, Y):
        """
        need to store:
        * V
        * sigmas
        * UU'
        * U'
        * V'

        if sliding window: only store the Y svd, as the X svd is the previous Y svd (store X only at start of traj).
        """
        current_covs = self.covs_[itraj]

        if self.mode == "sliding" and len(current_covs) > 0:
            U, S, V = self._U, current_covs[-1]._SS, current_covs[-1]._VV
        else:
            U, S, V_T = svds(X, k=self.k)
            V = V_T.T
        UU, SS, VV_T = svds(Y, k=self.k)
        VV = VV_T.T

        if self.mode == "sliding":
            self._U = UU

        current_covs.append(DecomposedCovPair(U, S, V, UU, SS, VV))

    def _estimate(self, X):
        self.covs_ = [[] for _ in range(X.ntraj)]

        if self.mode == 'sliding':
            splitter = SlidingCovariancesSplit(self.block_size)
        elif self.mode == 'linear':
            splitter = LinearCovariancesSplit(self.block_size)
        else:
            raise NotImplementedError("unsupported mode: %s" % self.mode)

        for data in splitter.split(X):
            self._process(*data)

        self.covs_ = np.array(self.covs_)

    # TODO: this would be an interface of tica, wouldnt it? (because it scores the estimated covariance matrix, cov or cov_tau?!)
    def score(self, percentage_test=0.30, scoring_method='vamp2'):
        """

        """
        if not self._estimated:
            raise RuntimeError('execute estimation first prior calling this method.')

        shape = self.covs_.shape
        p = (percentage_test, 1.0 - percentage_test)
        sample = np.random.choice((True, False), size=shape, p=p)

        test = self.covs_[sample]
        train = self.covs_[~sample]
        # split test and train test sets from input
        C00_test, C01_test, C11_test = test[0].combine(test[1:])
        C00_train, C01_train, C11_train = train[0].combine(train[1:])

        # TODO: why is this needed, eg. norming factor wrong?
        C00_test = (C00_test + C00_test.T) / 2.0
        C11_test = (C11_test + C11_test.T) / 2.0

        C00_train = (C00_train + C00_train.T) / 2.0
        C11_train = (C11_train + C11_train.T) / 2.0

        K = 'left singular values of Koopman operator'
        from pyemma.coordinates import covariance_lagged
        full_cov = covariance_lagged(self.data_producer, lag=1)
        from scipy.sparse.linalg import svds
        #from numpy.linalg import svd as svds
        #U, s, Vt = svds(full_cov.cov_tau, k=self.k)
        #K = U
        # K, C00_train, C0t_train, Ctt_train, C00_test, C0t_test, Ctt_test, k=None
        return vamp_score(K=full_cov.cov_tau, C00_test=C00_test, C0t_test=C01_test, Ctt_test=C11_test,
                          C00_train=C00_train, C0t_train=C01_train, Ctt_train=C11_train,
                          k=self.k, score=scoring_method)
