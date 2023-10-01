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



    def __call__(self, intg):

        if self.if_plot:
            self.plot_intg_cost(intg.costs[f'walltime'], 
                                name = self.plot_name, 
                                if_log = False)

        if self.outf:
            # Flatten the cost array
            # We need one entry per __call__
            # So eahc entry is a list separaterd with ','
            costs = intg.costs[f'walltime'].flatten()
            
            # Get the overall sum
            costs = sum(costs) 
            print(costs, 
                  sep=',',
                file=self.outf)

            # Flush to disk
            self.outf.flush()
