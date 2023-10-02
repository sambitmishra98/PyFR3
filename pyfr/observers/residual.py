import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.observers import BaseCost

class ResidualNorm(BaseCost):
    """
        # Residual in comparison with the steady-state solution
        # This is only applicable for implicit time integrators
        
    """
    name = 'res_l2'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

    def __call__(self, intg):
        super().__call__(intg)

        if self.outf:
            costs = intg.costs[self.cost_name].flatten()
            print(costs, sep=',', file=self.outf)
            self.outf.flush()
