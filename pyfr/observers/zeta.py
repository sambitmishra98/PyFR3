import numpy as np

from pyfr.observers import BaseParameter

class Zeta(BaseParameter):
    name = 'zeta'
    systems = ['ac-euler', 'ac-navier-stokes']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Initialise the output 
        def_zeta = self.cfg.getfloat('solver', 'ac-zeta')

        intg.parameters[self.name] = def_zeta * np.ones(
            (
             self._stages, 
             self._levels,
             self._pniters, 
            ))

        # Initialise with 1 to largest, to monitor indices
        #intg.parameters[self.name] = np.arange(
        #    1.0, 1.0 + stages*levels*pseudo_iterations).reshape(
        #        (stages, 
        #         levels,
        #         pseudo_iterations, 
        #        ))

    def __call__(self, intg):
        print('from parameters: ζ = ',intg.parameters)
