import numpy as np

from pyfr.mpiutil import get_comm_rank_root
from pyfr.optimisers.base import Cost, init_csv


class RuntimeCost(Cost):
    name = 'runtime'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        # Read configuration variables
        self.flushsteps = self.cfg.getint(self.cfgsect, 'flushsteps', 500)

        # MPI info
        comm, rank, root = get_comm_rank_root()

        if rank == root:
            self.outf = init_csv(self.cfg, cfgsect, 'n')
        else:
            self.outf = None

        self.costlist = np.zeros((1, 1))

    def __call__(self, intg):

        # Append performance info to the 1-D matrix
        self.costlist = np.append(self.costlist, intg.performanceinfo)

        print(self.process_cost(self.costlist))
