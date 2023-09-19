import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.observers import BaseCost

class WallTime(BaseCost):
    """
        # Residual in comparison with the steady-state solution
        # This is only applicable for implicit time integrators
        
    """
    name = 'walltime'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Initialise storage
        intg.costs[self.name] = np.ones((
                                        self._stages, 
                                        self._levels, 
                                        self._pniters,
                                         ))

    def __call__(self, intg):
        pass