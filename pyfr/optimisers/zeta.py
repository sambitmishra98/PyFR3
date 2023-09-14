import numpy as np

from pyfr.optimisers.base import BaseLocalOptimiser

class Zeta_optimiser(BaseLocalOptimiser):
    name = 'zeta_optimiser'
    systems = ['ac-euler','ac-navier-stokes']
    formulations = ['dual']

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

    def __call__(self, intg):
        super().__call__(intg)

        if self.cost[0] is None:
            return        

        # Get parameters list 
        print('from parameters: ζ = ',intg.parameters)
