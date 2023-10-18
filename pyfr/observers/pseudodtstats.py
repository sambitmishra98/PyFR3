import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.observers import BaseCost

class PseudoDtMin(BaseCost):
    name = 'dtau_min'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)
        self.comm, self.rank, self.root = get_comm_rank_root()

    def __call__(self, intg):
        super().__call__(intg)

        costs = intg.costs[self.cost_name].min(axis=0).flatten()
        self.comm.Allreduce(mpi.IN_PLACE, costs, op=mpi.MIN)
        if self.outf:
            print(*costs.tolist(), sep=',', file=self.outf)
            self.outf.flush()

class PseudoDtMean(BaseCost):
    name = 'dtau_mean'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)
        self.comm, self.rank, self.root = get_comm_rank_root()

    def __call__(self, intg):
        super().__call__(intg)

        costs = intg.costs[self.cost_name].mean(axis=0).flatten()
        n_ranks = self.comm.Get_size()
        self.comm.Allreduce(mpi.IN_PLACE, costs, op=mpi.SUM)
        costs /= n_ranks
        if self.outf:
            print(*costs.tolist(), sep=',', file=self.outf)
            self.outf.flush()
            print('done')

class PseudoDtMax(BaseCost):
    name = 'dtau_max'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)
        self.comm, self.rank, self.root = get_comm_rank_root()

    def __call__(self, intg):
        super().__call__(intg)

        costs = intg.costs[self.cost_name].max(axis=0).flatten()
        self.comm.Allreduce(mpi.IN_PLACE, costs, op=mpi.MAX)
        if self.outf:
            print(*costs.tolist(), sep=',', file=self.outf)
            self.outf.flush()