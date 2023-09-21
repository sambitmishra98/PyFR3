import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.observers import BaseCost

class SStResidualNorm(BaseCost):
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

        cost = intg.costs[self.cost_name]
                
        # Take diff across pseudo-iterations
        self.plot_intg_cost(cost, name = self.plot_name, if_log = False)
        
        # diff and cost are not the same shape
        # diff is (nsteps, nstages, nlevels, niter-1)
        # cost is (nsteps, nstages, nlevels, niter)
        # Pad diff with zeros along pseudo-iteration axis
        diff = np.diff(cost, axis = 2, append = 0)/cost
        self.plot_intg_cost(diff, name = self.plot_name+'_diff', if_log = False)
        
        # TODO
        # Create an iff condition that collects the sum across the step function in the pseudointegrator.
        # Remember, we ultimately need to decrease something like 
        #   (Δ𝓡/nΔτ)/Δ𝓣
        # where 
        #       Δ𝓡 is some norm of residual, 
        #       n  is some norm of the number of pseudo-iterations, and
        #       Δ𝓣 is the compute time
        
        # We will also need the ratio Δ𝓡/Δτ field for knowing which cycle is suitable where.

