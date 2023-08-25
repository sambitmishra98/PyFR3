from pyfr.mpiutil import get_comm_rank_root
from pyfr.optimisers.base import BaseLocalOptimiser, init_csv


class PrintCostOptimiser(BaseLocalOptimiser):
    name = 'printcost'
    systems = ['*']
    formulations = ['dual', 'std']

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

        print('initialised')

    def __call__(self, intg):

        print('called')


        # Write the current step number to the output file
        if self.outf:
            print('done', sep=',', file=self.outf)

            # Periodically flush to disk
            if intg.nacptsteps % self.flushsteps == 0:
                self.outf.flush()
