
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
'''
Created on 15.02.2016

@author: marscher
'''
from pyemma._base.serialization.serialization import SerializableMixIn


class Feature(SerializableMixIn):
    __serialize_version = 0
    __serialize_fields = ('dimension', 'top')

    @property
    def dimension(self):
        return self._dim

    @dimension.setter
    def dimension(self, val: int):
        self._dim = val

    @property
    def top(self):
        return self._top

    @top.setter
    def top(self, value):
        self._top = value

    def describe(self):
        raise NotImplementedError()

    def __eq__(self, other):
        if not isinstance(other, Feature):
            return False
        from pyemma.coordinates.data.featurization.util import hash_top
        return self.dimension == other.dimension and hash_top(self.top) == hash_top(other.top)

    def __repr__(self):
        return str(self.describe())
