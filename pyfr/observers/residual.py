import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.observers import BaseCost

class SStResidualNorm(BaseCost):
    """
        # Residual in comparison with the steady-state solution
        # This is only applicable for implicit time integrators
        
    """
    name = 'sst_residual_norm'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self.comm, self.rank, self.root = get_comm_rank_root()

        # Write the above if conditions in a more compact way
        self.index = {'p': 0, 'u': 1, 'v': 2, 'w': 3}[suffix]

        self.cost_name = self.name +'-'+ suffix# + '-' + str(self.lp)

        # Initialise storage
        intg.costs[self.cost_name] = np.zeros(
            (self._stages, self._levels, self._pniters))

    def __call__(self, intg):

        print(f'from costs: {self.name} = ',intg.costs)

