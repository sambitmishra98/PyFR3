import numpy as np

from pyfr.optimisers.base import BaseLocalOptimiser


class Zeta(BaseLocalOptimiser):
    name = 'zeta'
    systems = ['ac-euler','ac-navier-stokes']
    formulations = ['dual']

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

    def __call__(self, intg):
        super().__call__(intg)

        if self.cost[0] is None:
            return        

        # Get parameters list 
        print('all ζ: ',intg.parameters)

        print('ζ: ', intg.pseudointegrator.system.ac_zeta)
