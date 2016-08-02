import h5py
from pyemma._base.logging import Loggable
from pyemma.coordinates.data._base.datasource import DataSource
from pyemma.coordinates.data.data_in_memory import DataInMemoryIterator


class _CacheFileWrapper(Loggable):
    def __init__(self, data_producer, name):
        """
        Parameters
        ----------
        data_producer: DataSource

        file_handle : h5py.File

        """

        self._name = "pyemma.cache.fwrapper"
        self.data_producer = data_producer
        self.file_handle = self._cache_file = h5py.File(name=name, mode='a')

        if self.file_handle.items():
            self.validate_cache()

    def validate_cache(self):
        # check shapes
        for i, item in enumerate(self.file_handle.items()):
            desired = (self.data_producer.data_producer.trajectory_length(i), self.data_producer.ndim)
            if item[1].shape != desired:
                self.logger.debug("shape mismatch, removing: {} != {}".format(item[1].shape, desired))
                del self.file_handle[item[0]]
        self.file_handle.flush()

    def _create_cache_itraj(self, itraj):
        length = self.data_producer.trajectory_length(itraj)
        ndim = self.data_producer.ndim
        try:
            table = self.file_handle.create_dataset(name=str(itraj),
                                                    shape=(length, ndim),
                                                    dtype=self.data_producer.output_type())
            self.file_handle.flush()

        except:
            raise
        return table

    def fill_cache(self, table, itraj):
        # FIXME: this is gets much more than we actual want to assign here
        # howto solve this for non-random accessible formats?
        data_itraj = self.data_producer.data_producer.get_output()[itraj]
        table[:] = data_itraj
        self.file_handle.flush()
        return table

    def __getitem__(self, itraj):
        self.logger.debug("get itraj: {}".format(itraj))
        try:
            res = self.file_handle[str(itraj)]
            self.logger.debug("cache hit")
            return res
        except KeyError:
            self.logger.debug("cache miss")
            # 1. create table
            table = self._create_cache_itraj(itraj)

            # 2. fill cache
            self.fill_cache(table, itraj)

            return table


class _Cache(DataSource):
    """ This class caches the output of its data producer, respecting its stride value.

    TODO
    ----
    * how to ensure that we invalidate the cache in case of parameter changes
    * invalidate or switch file/dataset?
    *
    """

    def __init__(self, data_source, chunksize=1000):
        super(_Cache, self).__init__(chunksize=chunksize)
        self.data_producer = data_source

        self.data = _CacheFileWrapper(self, self.cache_name)

        self._ndim = self.data_producer.ndim
        self._ntraj = self.data_producer.ntraj
        self._lengths = self.data_producer.trajectory_lengths()

    @property
    def data_producer(self):
        return self._data_producer

    @data_producer.setter
    def data_producer(self, val):
        # TODO: this should invalidate the cache!
        self._data_producer = val

    @property
    def cache_name(self):
        """ file name of the cache"""
        # TODO: how to generate proper names automatically?
        return "{}".format("/tmp/test.h5")

    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=True, cols=None):
        return DataInMemoryIterator(self, skip, chunk, stride, return_trajindex, cols)


if __name__ == '__main__':
    import numpy as np
    import pyemma
    from glob import glob
    import os

    test_dir = "/tmp/foo"

    def create_test_file(force_recreate=False):

        if force_recreate or not os.path.exists(test_dir):
            print("creating")
            data = [np.random.random((1000, 12)) for _ in range(10)]
            os.makedirs(test_dir)
            os.chdir(test_dir)
            for i, x in enumerate(data):
                np.save("{}.npy".format(i), x)

    create_test_file()

    files = glob(test_dir+"/*.npy")
    print("files", files)

    src = pyemma.coordinates.source(files)
    data = src.get_output()
    cache = _Cache(src)

    out = cache.get_output()

    for actual, desired in zip(out, data):
        np.testing.assert_allclose(actual, desired, atol=1e-15, rtol=1e-6)
