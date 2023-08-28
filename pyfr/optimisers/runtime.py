import numpy as np

from pyfr.optimisers.base import Cost


class RuntimeCost(Cost):
    name = 'runtime'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        self.costlist = np.zeros((1, 1))

    def __call__(self, intg):

        # Append performance info to the 1-D matrix
        self.costlist = np.append(self.costlist, intg.performanceinfo)
