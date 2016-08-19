from collections import defaultdict
import os
import h5py
import numpy as np

from pyemma._base.logging import Loggable
from pyemma._base.progress import ProgressReporter
from pyemma.coordinates.data._base.datasource import DataSource
from pyemma.coordinates.data.data_in_memory import DataInMemoryIterator
from pyemma.coordinates.data.util.traj_info_cache import TrajectoryInfoCache
from pyemma.util import config
from pyemma.util.annotators import fix_docs
from pyemma.util.units import bytes_to_string


class _CacheFile(Loggable, ProgressReporter):
    def __init__(self, cache, name):
        """
        Parameters
        ----------
        cache: DataSource
            the associated Cache instance to which this _CacheFile belongs to.

        name: str
            file name of the underlying cache file.

        """

        self._name = "pyemma.cache.fwrapper"
        self.cache = cache
        self.data_source = cache.data_producer
        self.file_name = os.path.join(config.cache_dir, name)
        try:
            self.file_handle = self._cache_file = h5py.File(name=self.file_name, mode='a')
        except:
            raise
        self.misses = 0
        self.hits = defaultdict(int)
        if self.file_handle.items():
            self.validate_cache()

    def validate_cache(self):
        # 0. we've ensured that different parameters or pipeline arrangement will use another cache file!
        # 1. check shapes, remove those which do not match.
        # 2.
        for i, item in enumerate(self.file_handle.items()):
            desired = (self.data_source.trajectory_length(i), self.cache.ndim)
            if item[1].shape != desired:
                self.logger.debug("shape mismatch, removing: {} != {}".format(item[1].shape, desired))
                del self.file_handle[item[0]]
        self.file_handle.flush()

    def _create_cache_itraj(self, traj_info):
        length = traj_info.length
        ndim = traj_info.ndim
        try:
            dataset = self.file_handle.create_dataset(name=traj_info.hash_value,
                                                      shape=(length, ndim),
                                                      dtype=self.cache.output_type(),
                                                      chunks=True,
                                                      )
            self.file_handle.flush()

        except:
            raise
        return dataset

    def fill_cache(self, table, itraj):
        t = 0
        with self.data_source.iterator(chunk=self.data_source.chunksize) as it:
            it.state.itraj = itraj
            if self.data_source.chunksize:
                n_chunks_for_itraj = int(self.data_source.trajectory_length(itraj) / self.data_source.chunksize)
                self._progress_register(n_chunks_for_itraj, description="fill cache for traj={}".format(itraj))
            for itraj_iter, chunk in it:
                if itraj_iter != itraj:
                    break
                n = len(chunk)
                table[t:t + n] = chunk[:]
                t += n
                if self.data_source.chunksize:
                    self._progress_update(1)

        if self.data_source.chunksize:
            self._progress_force_finish()

        self.file_handle.flush()
        return table

    def _itraj_to_file_hash(self, itraj):
        file = self.cache.filenames[itraj]
        inst = TrajectoryInfoCache.instance()
        info = inst[file, self.cache._real_reader]
        return info

    def __getitem__(self, itraj):
        """
        fake a list of arrays and give underlying h5py dataset containing the real data.

        Parameters
        ----------
        itraj

        Returns
        -------
        table

        """
        traj_info = self._itraj_to_file_hash(itraj)
        try:
            res = self.file_handle[traj_info.hash_value]
            self.hits[itraj] += 1
            self.logger.debug("HIT")
            return res
        except KeyError:
            self.logger.debug("MISS")
            self.misses += 1
            # 1. create table
            table = self._create_cache_itraj(traj_info)

            # 2. fill cache
            self.fill_cache(table, itraj)

            return table

    def __repr__(self):
        size = bytes_to_string(self.file_handle.id.get_filesize())
        return "[CacheFile {fn}: items={n} size={size}]".format(fn=self.file_handle.filename,
                                                                n=len(self.file_handle),
                                                                size=size)

    def __len__(self):
        return len(self.file_handle)


@fix_docs
class _Cache(DataSource):
    """ This class caches the output of its data producer

    TODO
    ----
    * how to ensure that we invalidate the cache in case of parameter changes
    * invalidate or switch file/dataset?
    *
    """

    def __init__(self, data_source, chunksize=1000):
        super(_Cache, self).__init__(chunksize=chunksize)

        self.data_producer = data_source
        reader = self.data_producer
        while not reader.is_reader:
            reader = reader.data_producer

        assert reader
        assert reader.is_reader
        self._real_reader = reader

        first_cache_file = _CacheFile(self, self.current_cache_file_name)

        # provide the data list
        self._cache_files = {self.current_cache_file_name: first_cache_file}

        self._ndim = self.data_producer.ndim
        self._ntraj = self.data_producer.ntraj
        self._lengths = self.data_producer.trajectory_lengths()

    @property
    def data(self):
        return self._cache_files[self.current_cache_file_name]

    @property
    def data_producer(self):
        return self._data_producer

    @data_producer.setter
    def data_producer(self, val):
        # TODO: this should invalidate the cache!
        # self.data.invalidate()
        self._data_producer = val

    @property
    def current_cache_file_name(self):
        """ file name of the cache.

        it is unique depending on the pipeline structure and the estimation parameters of each
        element in the pipeline

        """
        descriptions = []

        from pyemma.coordinates.data import FeatureReader
        if isinstance(self._real_reader, FeatureReader):
            descriptions.append(self._real_reader.featurizer.describe())

        # get params of estimators
        dp = self.data_producer
        while dp is not dp.data_producer:
            from pyemma._base.estimator import Estimator
            if isinstance(dp, Estimator):
                descriptions.append(repr(dp))

            from pyemma.coordinates.transform.transformer import StreamingTransformer
            if isinstance(dp, StreamingTransformer):
                pass
            dp = dp.data_producer

        # TODO: we maybe do not have a pipeline, but just a reader without descriptions, but we do not want to include the filenames, right?
        if not descriptions:
            descriptions.append(self._real_reader.filenames)

        #import pprint
        #self.logger.debug("descriptions:\n{}".format(pprint.pformat(descriptions)))

        # hash description list to build file name
        from hashlib import sha256
        hasher = sha256()
        inp = str(descriptions).encode()
        hasher.update(inp)
        res = hasher.hexdigest()
        return "{prefix}_{hash}.{suffix}".format(prefix=self._data_producer.__class__.__name__,
                                                 hash=res, suffix="hdf5")

    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=True, cols=None):
        return DataInMemoryIterator(self, skip, chunk, stride, return_trajindex, cols)

    def get_output(self, dimensions=slice(0, None), stride=1, skip=0, chunk=None):
        if isinstance(dimensions, int):
            ndim = 1
            dimensions = slice(dimensions, dimensions + 1)
        elif isinstance(dimensions, (list, np.ndarray, tuple, slice)):
            if hasattr(dimensions, 'ndim') and dimensions.ndim > 1:
                raise ValueError('dimension indices can\'t have more than one dimension')
            ndim = len(np.zeros(self.ndim)[dimensions])
        else:
            raise ValueError('unsupported type (%s) of "dimensions"' % type(dimensions))
        assert ndim > 0, "ndim was zero in %s" % self.__class__.__name__
        dim_inds = dimensions.indices(1)
        self.logger.debug("dim slice: {}; dim inds: {}".format(dimensions, dim_inds))

        res = [traj[skip::stride, dimensions] for traj in self.data]
        assert all(isinstance(o, np.ndarray) for o in res)
        return res

    def __len__(self):
        return sum(len(f) for f in self._cache_files)

    def __repr__(self):
        # TODO: add a description of the current pipeline
        return "[Cache => {}".format(repr(self._cache_files[self.current_cache_file_name]))
