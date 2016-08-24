import tempfile
import unittest
import numpy as np
import pyemma
from glob import glob
import os

from pyemma.coordinates.data.cache import _Cache
from pyemma.util import config

class ProfilerCase(unittest.TestCase):
    def setUp(self):
        import cProfile
        self.pr = cProfile.Profile()
        self.pr.enable()

    def tearDown(self):
        self.pr.dump_stats(self._testMethodName)


class TestCache(ProfilerCase):
    
    @classmethod
    def setUpClass(cls):
        config.use_trajectory_lengths_cache = False
        cls.test_dir = tempfile.mkdtemp(prefix="test_cache_")

        cls.length = 1000
        cls.dim = 3
        data = [np.random.random((cls.length, cls.dim)) for _ in range(10)]

        for i, x in enumerate(data):
            np.save(os.path.join(cls.test_dir, "{}.npy".format(i)), x)

        cls.files = glob(cls.test_dir + "/*.npy")
    
    def setUp(self):
        super(TestCache, self).setUp()
        self.tmp_cache_dir = tempfile.mkdtemp(dir=self.test_dir)
        config.cache_dir = self.tmp_cache_dir

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.test_dir, ignore_errors=True)

        from pyemma.coordinates.data.cache import _used_files
        import pprint
        pprint.pprint(_used_files)
        print("*"*80)
        pprint.pprint(set(_used_files))

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

    #@unittest.skip("test")
    def test_tica_cached_input(self):
        src = pyemma.coordinates.source(self.files, chunk_size=0)
        cache = _Cache(src)
        print("cache inp fileeeeeeeeeeeeee", cache.current_cache_file_name)

        tica_cache_inp = pyemma.coordinates.tica(cache)

        tica_without_cache = pyemma.coordinates.tica(src)

        np.testing.assert_allclose(tica_cache_inp.cov, tica_without_cache.cov, atol=1e-10)
        np.testing.assert_allclose(tica_cache_inp.cov_tau, tica_without_cache.cov_tau, atol=1e-9)

        np.testing.assert_allclose(tica_cache_inp.eigenvalues, tica_without_cache.eigenvalues, atol=1e-7)
        np.testing.assert_allclose(np.abs(tica_cache_inp.eigenvectors),
                                   np.abs(tica_without_cache.eigenvectors), atol=1e-6)

    def test_tica_cached_output(self):
        src = pyemma.coordinates.source(self.files, chunk_size=0)
        tica = pyemma.coordinates.tica(src)

        tica_output = tica.get_output()
        cache = _Cache(tica)
        print("cache fileeeeeeeeeeeeee", cache.current_cache_file_name)

        np.testing.assert_allclose(cache.get_output(), tica_output)

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

        # remove 2nd feature and check we've got the old name back.
        reader.featurizer.active_features.pop()
        self.assertEqual(cache.current_cache_file_name, name_of_cache)

        # add a new file and ensure we still got the same cache file
        reader.filenames.append(os.path.join(path, 'bpti_001-033.xtc'))
        self.assertEqual(cache.current_cache_file_name, name_of_cache)



if __name__ == '__main__':
    unittest.main()
