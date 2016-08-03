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

        self.length = 10000
        self.dim = 3
        data = [np.random.random((self.length, self.dim)) for _ in range(10)]
        os.chdir(self.test_dir)
        for i, x in enumerate(data):
            np.save("{}.npy".format(i), x)

        self.files = glob(self.test_dir + "/*.npy")
        #pyemma.config.show_progress_bars = True

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cache_hits(self):
        src = pyemma.coordinates.source(self.files, chunk_size=1000)
        src.describe()
        data = src.get_output()
        cache = _Cache(src)

        out = cache.get_output()
        self.assertEqual(cache.data.misses, len(self.files))

        for actual, desired in zip(out, data):
            np.testing.assert_allclose(actual, desired, atol=1e-15, rtol=1e-7)

        cache.get_output()
        self.assertEqual(len(cache.data.hits), len(self.files))

        self.assertIn("items={}".format(len(cache)), repr(cache))

    def test_with_tica(self):
        src = pyemma.coordinates.source(self.files, chunk_size=0)
        cache = _Cache(src)
        tica_cache_inp = pyemma.coordinates.tica(cache)
        print(tica_cache_inp.dimension())

        tica_without_cache = pyemma.coordinates.tica(src)

        np.testing.assert_allclose(tica_cache_inp.cov, tica_without_cache.cov, atol=1e-10)
        np.testing.assert_allclose(tica_cache_inp.cov_tau, tica_without_cache.cov_tau, atol=1e-10)

        np.testing.assert_allclose(tica_cache_inp.eigenvalues, tica_without_cache.eigenvalues, atol=1e-8)
        np.testing.assert_allclose(tica_cache_inp.eigenvectors, tica_without_cache.eigenvectors, atol=1e-8)


    def test_with_tica_withoutcache(self):
        src = pyemma.coordinates.source(self.files, chunk_size=0)
        #cache = _Cache(src)
        tica = pyemma.coordinates.tica(src)