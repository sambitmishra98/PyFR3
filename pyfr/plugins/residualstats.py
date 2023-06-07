import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolnPlugin, init_csv


class ResidualStatsPlugin(BaseSolnPlugin):
    name = 'residualstats'
    systems = ['*']
    formulations = ['std', 'dual']

    def __init__(self, intg, cfgsect, prefix):
        super().__init__(intg, cfgsect, prefix)

        self.flushsteps = self.cfg.getint(self.cfgsect, 'flushsteps', 500)
        self.stats = []
        self.t_prev = intg.tcurr
        self.fvars = intg.system.elementscls.convarmap[self.ndims]
        self.ele_types = intg.system.ele_types

        # Maximum of 3 Levels of abstraction for the stats of pseudo-dt field
        self.res_stats = { 
            'n'  : {'all':0},
            'res': {'all':0}|{p:{'all':0} for p in self.fvars}, 
            }

        if 'solver-dual-time-integrator-multip' in intg.cfg.sections():
            self.level = self.cfg.getint(self.cfgsect, 
                'level', intg.cfg.getint('solver','order'))

        csv_header =  'pseudo-steps, tcurr'
        for k, v in self.res_stats.items():
            if k != 'all':
                csv_header += f',{k}'
                for vk, vv in v.items():
                    if vk != 'all':
                        csv_header += f',{k}_{vk}'

        # MPI info
        self.comm, self.rank, self.root = get_comm_rank_root()

        # The root rank needs to open the output file
        if (self.rank == self.root):
            self.outf = init_csv(self.cfg, cfgsect, csv_header)
        else:
            self.outf = None

        self.stored_t_prev = None
        self.last_appendable_stat = None

    def __call__(self, intg):
        # Process the sequence of pseudo-residuals

        for (npiter, iternr, resid) in intg.pseudostepinfo:

            if iternr == 1:           # We can store the last step's data
                if self.last_appendable_stat != None:
                    if self.stored_t_prev != self.t_prev:
                        self.stats.append((f for f in self.last_appendable_stat))
                        self.prev_npiter = npiter - 1
                else:
                    self.prev_npiter = 0 

                self.stored_t_prev = self.t_prev

            self.res_stats['n']['all'] = npiter-self.prev_npiter

            self.residual_statistics(intg, resid)
            self.last_appendable_stat = (npiter, intg.tcurr, 
                                    *self.res_stats_as_list(self.res_stats))

        self.t_prev = intg.tcurr

        # If we're the root rank then output
        if self.outf:
            for s in self.stats:
                print(*s, sep=f',', file=self.outf)

            # Periodically flush to disk
            if intg.nacptsteps % self.flushsteps == 0:
                self.outf.flush()

        # Reset the stats
        self.stats = []

    def residual_statistics(self, intg, resid):
        '''
            Use a list of numpy arrays, one for each element type.
            Each array is of shape(nvars,)
        '''
        resid = resid or (0,)*intg.system.nvars

        # each variable in (p, u, v, w)
        for j, var in enumerate(self.fvars):
            self.res_stats['res'][var]['all'] = resid[j]

        pseudo_resid_norm = self.cfg.get('solver-time-integrator', 'pseudo-resid-norm')

        if pseudo_resid_norm == 'uniform':
            self.res_stats['res']['all'] = max(self.res_stats['res'][var]['all'] for var in self.fvars)
        elif pseudo_resid_norm in ['l2', 'l4', 'l8']:
            self.res_stats['res']['all'] = sum(self.res_stats['res'][var]['all'] for var in self.fvars)
        else:
            raise ValueError(f'Unknown time integrator: {pseudo_resid_norm}')

    def __FUTURE_residual_statistics(self, resid):
        '''
            Use a list of numpy arrays, one for each element type.
            Each array is of shape(nvars,)
        '''

        for j, var in enumerate(self.fvars):
            for i, e_type in enumerate(self.ele_types):

                # each element type, each variable in (p, u, v, w)
                # Stats obtained over all elements
                self.res_stats['res'][var][e_type]['each'] = resid[i][j]

                # each element type, each variable in (p, u, v, w)
                # Stats obtained over all elements and element soln points

                res = np.array(self.res_stats['res'][var][e_type]['each'].max())

                if self.rank != self.root: self.comm.Reduce(res, None , op=mpi.SUM, root = self.root)
                else:                      self.comm.Reduce(mpi.IN_PLACE, res, op=mpi.SUM, root = self.root)

                self.res_stats['res'][var][e_type]['all'] = res

            # each variable in (p, u, v, w)
            # Stats obtained over all element types, elements and element soln points
            self.res_stats['res'][var]['all'] = max([self.res_stats['res'][var][e_type]['all'] for e_type in self.ele_types])

        # Stats obtained 
        #   over all element types, elements, variable 
        #   in (p, u, v, w) 
        #   and element soln points
        self.res_stats['res']['all'] = max([self.res_stats['res'][var]['all'] 
                                           for var in self.fvars])

    def res_stats_as_list(self, Δτ_stats):
        Δτ_stats_list = []
        for v in Δτ_stats.values():
            Δτ_stats_list.append(v['all'])               
            for vk, vv in v.items():
                if isinstance(vv, dict):    
                    Δτ_stats_list.append(vv['all'])

        return Δτ_stats_list
    