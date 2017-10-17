import random


class FixedSeedMixIn(object):

    @property
    def fixed_seed(self):
        """ seed for random choice of initial cluster centers. Fix this to get reproducible results."""
        return self._fixed_seed

    @fixed_seed.setter
    def fixed_seed(self, val):
        from pyemma.util import types
        if isinstance(val, bool) or val is None:
            if val:
                self._fixed_seed = 42
            else:
                self._fixed_seed = random.randint(0, 2 ** 32 - 1)
        elif types.is_int(val):
            if val < 0 or val > 2 ** 32 - 1:
                if hasattr(self, 'logger'):
                    self.logger.warn("seed has to be positive (or smaller than 2**32-1)."
                                     " Seed will be chosen randomly.")
                self.fixed_seed = False
            else:
                self._fixed_seed = val
        else:
            raise ValueError("fixed seed has to be bool or integer")

        if hasattr(self, 'logger'):
            self.logger.debug("seed = %i", self._fixed_seed)
