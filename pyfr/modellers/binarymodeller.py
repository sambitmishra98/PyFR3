import numpy as np

from pyfr.modellers.base import BaseLocalModeller

class BinaryModeller(BaseLocalModeller):
    name = 'binarymodeller'
    systems = ['*']
    formulations = ['std', 'dual']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Name of the parameter?
        self.parameter_name = 'zeta'
        self.cost_name = 'sst_residual_norm-p'

        self.parameter = intg.parameters[self.parameter_name]

    def __call__(self, intg):

        # Process the intg.costs dictionary
        intg.model = self.postprocess(intg.costs['sst_residual_norm-p'])

        # Clean intg.costs for the next time step as an array of zeros
        intg.costs = {self.cost_name: np.zeros(
            (self._stages, self._levels, self._pniters))}

    def postprocess(self, residual):
        rel = np.zeros(
            (self._stages, self._levels, self._pniters))

        # We need ΔR/R = (Rₙ - Rₙ₋₁)/Rₙ₋₁ along the pseudo_iterations axis
        rel[:,:,1:] = (residual[:,:,1:] - residual[:,:,:-1])/residual[:,:,:-1]

        rel[rel > 0] = +1
        rel[rel < 0] = -1

        return rel
