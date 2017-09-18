import numpy as np

from pyemma._base.serialization.serialization import SerializableMixIn

class test_cls_v1(SerializableMixIn):
    _serialize_fields = ('a', 'x', 'y')
    _serialize_version = 1

    def __init__(self):
        self.a = np.array([1, 2, 3])
        self.x = 'foo'
        self.y = 0.0

    def __eq__(self, other):
        return np.allclose(self.a, other.a) and self.x == other.x and self.y == other.y


# note: in the real world this decorator would be invoked here!
#@handle_old_classes('pyemma._base.serialization.tests.test_serialization.test_cls_v1')
class test_cls_v2(SerializableMixIn):
    _serialize_fields = ('b', 'y', 'z')
    _serialize_version = 2
    # interpolate from version 1: add attr z with value 42
    _serialize_interpolation_map = {1: [('set', 'z', 42),
                                        ('mv', 'a', 'b'),
                                        ('rm', 'x')]}

    def __init__(self):
        self.b = np.array([1, 2, 3])
        self.y = 0.0
        self.z = 42

    def __eq__(self, other):
        return np.allclose(self.b, other.b) and self.y == other.y and self.z == other.z


class test_cls_v3(SerializableMixIn):
    # should fake the refactoring of new_cls
    _serialize_fields = ('c', 'z')
    _serialize_version = 3
    # interpolate from version 1 and 2
    _serialize_interpolation_map = {1: [('set', 'z', 42),
                                        ('mv', 'a', 'b'),
                                        ('rm', 'x')],
                                    2: [('set', 'z', 23),
                                        ('mv', 'b', 'c'),
                                        ('rm', 'y')]}

    def __init__(self):
        self.c = np.array([1, 2, 3])
        self.z = 23

    def __eq__(self, other):
        return np.allclose(self.c, other.c) and self.y == other.y and self.z == other.z


class _deleted_in_old_version(test_cls_v3):
    _serialize_version = 4

    def __init__(self):
        super(_deleted_in_old_version, self).__init__()


class test_cls_with_old_locations(_deleted_in_old_version):
    _serialize_version = 5

    def __init__(self):
        super(test_cls_with_old_locations, self).__init__()


class to_interpolate_with_functions(test_cls_v1):

    @staticmethod
    def map_y(x):
        return 42

    _serialize_version = 2
    # map from version 1 to 2
    _serialize_interpolation_map = {1: [('map', 'y', map_y)]}