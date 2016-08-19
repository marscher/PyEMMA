import tempfile
import unittest
import numpy as np
import pyemma
from glob import glob
import os

from pyemma.coordinates.data.cache import _Cache


class TestCache(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        self.length = 1000
        self.dim = 3
        data = [np.random.random((self.length, self.dim)) for _ in range(10)]

        for i, x in enumerate(data):
            np.save(os.path.join(self.test_dir, "{}.npy".format(i)), x)

        self.files = glob(self.test_dir + "/*.npy")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cache_hits(self):
        src = pyemma.coordinates.source(self.files, chunk_size=1000)
        src.describe()
        data = src.get_output()
        cache = _Cache(src)
        self.assertEqual(cache.data.misses, 0)
        out = cache.get_output()
        self.assertEqual(cache.data.misses, len(self.files))

        for actual, desired in zip(out, data):
            np.testing.assert_allclose(actual, desired, atol=1e-15, rtol=1e-7)

        cache.get_output()
        self.assertEqual(len(cache.data.hits), len(self.files))

        #self.assertIn("items={}".format(len(cache)), repr(cache))

    def test_get_output(self):
        src = pyemma.coordinates.source(self.files, chunk_size=0)
        dim = 1
        stride = 2
        skip = 3
        desired = src.get_output(stride=stride, dimensions=dim, skip=skip)

        cache = _Cache(src)
        actual = cache.get_output(stride=stride, dimensions=dim, skip=skip)
        np.testing.assert_allclose(actual, desired)

    def test_with_tica(self):
        src = pyemma.coordinates.source(self.files, chunk_size=0)
        cache = _Cache(src)
        tica_cache_inp = pyemma.coordinates.tica(cache)

        tica_without_cache = pyemma.coordinates.tica(src)

        np.testing.assert_allclose(tica_cache_inp.cov, tica_without_cache.cov, atol=1e-10)
        np.testing.assert_allclose(tica_cache_inp.cov_tau, tica_without_cache.cov_tau, atol=1e-9)

        np.testing.assert_allclose(tica_cache_inp.eigenvalues, tica_without_cache.eigenvalues, atol=1e-7)
        np.testing.assert_allclose(np.abs(tica_cache_inp.eigenvectors),
                                   np.abs(tica_without_cache.eigenvectors), atol=1e-6)

    def test_cache_switch_cache_file(self):
        src = pyemma.coordinates.source(self.files, chunk_size=0)
        t = pyemma.coordinates.tica(src)
        cache = _Cache(t)

    def test_with_feature_reader_switch_cache_file(self):
        import pkg_resources
        path = pkg_resources.resource_filename(__name__, 'data') + os.path.sep
        pdbfile = os.path.join(path, 'bpti_ca.pdb')
        trajfiles = os.path.join(path, 'bpti_mini.xtc')

        reader = pyemma.coordinates.source(trajfiles, top=pdbfile)
        reader.featurizer.add_selection([0,1, 2])

        cache = _Cache(reader)
        name_of_cache = cache.current_cache_file_name

        reader.featurizer.add_selection([5, 8, 9])
        new_name_of_cache = cache.current_cache_file_name

        self.assertNotEqual(name_of_cache, new_name_of_cache)

if __name__ == '__main__':
    unittest.main()
