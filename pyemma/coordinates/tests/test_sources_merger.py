import unittest
import pkg_resources
import os
from glob import glob
import numpy as np

from pyemma.coordinates import source
from pyemma.coordinates.data.sources_merger import SourcesMerger
from pyemma import config


class TestSourcesMerger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config.coordinates_check_output = True
        import tempfile
        cls.testdir = tempfile.mkdtemp()
        cls.ntrajs = 4
        cls.dims = [np.random.randint(10, 100) for _ in range(cls.ntrajs//2)]*2
        cls.traj_lengths = 100
        cls.npy_files = []
        cls.data = []
        for i, dim in enumerate(cls.dims):
            data = np.random.random((cls.traj_lengths, dim))
            fn = os.path.join(cls.testdir, '%s.npy' % i)
            np.save(fn, data)
            cls.npy_files.append(fn)
            cls.data.append(data)

    @classmethod
    def tearDownClass(cls):
        config.coordinates_check_output = False
        import shutil
        shutil.rmtree(cls.testdir)

    def setUp(self):
        self.readers = []
        self.readers.append(source(self.npy_files[:2]))
        self.readers.append(source(self.npy_files[2:]))
        self.desired_combined_output = None

    def _get_output_compare(self, joiner, stride=1, chunk=0, skip=0):
        j = joiner
        out = j.get_output(stride=stride, chunk=chunk, skip=skip)
        assert len(out) == 3
        assert j.ndim == self.readers[0].ndim * 2
        np.testing.assert_equal(j.trajectory_lengths(), self.readers[0].trajectory_lengths())

        from collections import defaultdict
        outs = defaultdict(list)
        for r in self.readers:
            for i, x in enumerate(r.get_output(stride=stride, chunk=chunk, skip=skip)):
                outs[i].append(x)
        combined = [np.hstack(outs[i]) for i in range(3)]
        np.testing.assert_equal(out, combined)

    def test_combined_output(self):
        j = SourcesMerger(self.readers)
        self._get_output_compare(j, stride=1, chunk=0, skip=0)
        self._get_output_compare(j, stride=2, chunk=5, skip=0)
        self._get_output_compare(j, stride=2, chunk=13, skip=3)
        self._get_output_compare(j, stride=3, chunk=2, skip=7)

    def test_ra_stride(self):
        ra_indices = np.array([[0,7], [0, 23], [1, 30], [2, 9]])
        j = SourcesMerger(self.readers)

        self._get_output_compare(j, stride=ra_indices)

    def test_non_matching_lengths(self):
        data = self.readers[1].data
        data = [data[0], data[1], data[2][:20]]
        self.readers.append(source(data))
        with self.assertRaises(ValueError) as ctx:
            SourcesMerger(self.readers)
        self.assertIn('matching', ctx.exception.args[0])

    def test_fragmented_trajs(self):
        """ build two fragmented readers consisting out of two fragments each and check if they are merged properly."""
        segment_0 = np.arange(20)
        segment_1 = np.arange(20, 40)

        s1 = source([(segment_0, segment_1)])
        s2 = source([(segment_0, segment_1)])

        sm = SourcesMerger((s1, s2))

        out = sm.get_output()
        x = np.atleast_2d(np.arange(40))
        expected = [np.concatenate((x, x), axis=0).T]

        np.testing.assert_equal(out, expected)
