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
        
        # Norm used on residual
        self.lp = self.cfg.getfloat(cfgsect, 'norm', 2)

        # Set MPI reduction op and post process function
        if self.lp == float('inf'):
            self._mpi_op = mpi.MAX
            self._post_func = lambda x: x
        else:
            self._mpi_op = mpi.SUM
            self._post_func = lambda x: x**(1/self.lp)

        self.cost_name = self.name +'-'+ suffix + '-' + self.lp

        # Initialise storage
        intg.costs[self.cost_name] = np.ones(
            (self._stages, self._levels, self._pniters))

    def __call__(self, intg):

        norm = lambda x: np.linalg.norm(x, axis=(0, 2), ord=self.lp)
        if self.lp == float('inf'):
            resid = max(norm(dt_s) for dt_s in intg.dt_soln)
        else:
            resid = sum(norm(dt_s)**self.lp for dt_s in intg.dt_soln)

        # Reduce and, if we are the root rank, output
        if self.rank != self.root:
            self.comm.Reduce(resid, None, op=self._mpi_op, 
                             root=self.root)
        else:
            self.comm.Reduce(mpi.IN_PLACE, resid, op=self._mpi_op, 
                             root=self.root)

            # Post process
            resid = list(self._post_func(r) for r in resid)
            print(f'L₂ norm of {self.suffix}: ', resid[self.index])

        print(f'from costs: {self.name} = ',intg.costs)
