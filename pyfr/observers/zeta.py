import numpy as np

from pyfr.observers import BaseParameter

class Zeta(BaseParameter):
    name = 'zeta'
    systems = ['ac-euler', 'ac-navier-stokes']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        if intg.parameters.get(self.name) is None:

            z = self.cfg.getfloat('solver', 'ac-zeta')
            intg.parameters[self.name] = z * np.ones(
                (
                self._stages, 
                self._levels,
                self._pniters, 
                ))

        print(f"NEW: {intg.parameters[self.name]}")

        # ----------------------------------------------------------------------

    def __call__(self, intg):
        return
