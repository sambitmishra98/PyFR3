import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import BaseSolnPlugin, init_csv


class PseudodtStatsPlugin(BaseSolnPlugin):
    name = 'pseudodt'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, prefix):
        super().__init__(intg, cfgsect, prefix)

        self.flushsteps = self.cfg.getint(self.cfgsect, 'flushsteps', 500)

        self.stats = []

        self.t_prev = intg.tcurr

        self.fvars = intg.system.elementscls.convarmap[self.ndims]
        self.ele_types = intg.system.ele_types

        # Pseudo-dt field can be abstracted to 4 levels: overall, field-variables, element types and multi-p levels
        # TODO: Implement abstraction for multi-p levels
        self.Δτ_field_stats = { 
            'n'  : {'all':0},
            'min': {'all':0}|{p:{'all':0}|{e:{'all':0} for e in self.ele_types} for p in self.fvars}, 
            'max': {'all':0}|{p:{'all':0}|{e:{'all':0} for e in self.ele_types} for p in self.fvars},
                        }

        if 'solver-dual-time-integrator-multip' in intg.cfg.sections():
            self.level = self.cfg.getint(self.cfgsect, 
                'level', intg.cfg.getint('solver','order'))

        csv_header =  'pseudo-steps, tcurr'
        for k, v in self.Δτ_field_stats.items():
            if k != 'all':
                csv_header += f',{k}'
                for vk, _ in v.items():
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

        for (npiter, iternr, _) in intg.pseudostepinfo:

            if iternr == 1:           # We can store the last step's data

                # TODO: Remove this check
                if 'solver-dual-time-integrator-multip' in intg.cfg.sections():
                    Δτ_field = intg.pseudointegrator.pintgs[self.level].dtau_mats
                else:
                    Δτ_field = intg.pseudointegrator.dtau_mats
                
                if self.last_appendable_stat != None:
                    if self.stored_t_prev != self.t_prev:
                        self.stats.append((f for f in self.last_appendable_stat))
                        self.prev_npiter = npiter - 1
                else:
                    self.prev_npiter = 0 

                self.stored_t_prev = self.t_prev

            self.Δτ_field_stats['n']['all'] = npiter-self.prev_npiter

            self.Δτ_statistics(Δτ_field)
            self.last_appendable_stat = (npiter, intg.tcurr, 
                                    *self.Δτ_stats_as_list(self.Δτ_field_stats))

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

    def Δτ_statistics(self, Δτ_field):
        '''
            self.Δτ_field is a list of matrices, one for each element type.
            Each matrix is a 3D array of shape (nupts, nvars, neles)
        '''

        for j, var in enumerate(self.fvars):
            for i, e_type in enumerate(self.ele_types):

                # each element type, each soln point in element, each variable in (p, u, v, w)
                # Stats obtained over all elements
                self.Δτ_field_stats['min'][var][e_type]['each'] = (Δτ_field[i][:, j, :].min(1))
                self.Δτ_field_stats['max'][var][e_type]['each'] = (Δτ_field[i][:, j, :].max(1))

                # each element type, each variable in (p, u, v, w)
                # Stats obtained over all elements and element soln points
                self.Δτ_field_stats['min'][var][e_type]['all'] = (self.Δτ_field_stats['min'][var][e_type]['each'].min())
                self.Δτ_field_stats['max'][var][e_type]['all'] = (self.Δτ_field_stats['max'][var][e_type]['each'].max())

            # each variable in (p, u, v, w)
            # Stats obtained over all element types, elements and element soln points
            self.Δτ_field_stats['min'][var]['all'] = min([self.Δτ_field_stats['min'][var][e_type]['all'] for e_type in self.ele_types])
            self.Δτ_field_stats['max'][var]['all'] = max([self.Δτ_field_stats['max'][var][e_type]['all'] for e_type in self.ele_types])

            t_max = np.array(self.Δτ_field_stats['min'][var]['all'])
            t_min = np.array(self.Δτ_field_stats['max'][var]['all'])

            if self.rank != self.root:
                self.comm.Reduce(t_max, None, op=mpi.MIN, root=self.root)
                self.comm.Reduce(t_min, None, op=mpi.MAX, root=self.root)
            else:
                self.comm.Reduce(mpi.IN_PLACE, t_max, op=mpi.MIN, root=self.root)
                self.comm.Reduce(mpi.IN_PLACE, t_min, op=mpi.MAX, root=self.root)

            self.Δτ_field_stats['min'][var][e_type]['all'] = t_max
            self.Δτ_field_stats['max'][var][e_type]['all'] = t_min

        # Stats obtained over 
        #   all element types, elements, variable 
        #   in (p, u, v, w) 
        #   and element soln points
        self.Δτ_field_stats['min']['all'] = min([self.Δτ_field_stats['min'][var]['all'] for var in self.fvars])
        self.Δτ_field_stats['max']['all'] = max([self.Δτ_field_stats['max'][var]['all'] for var in self.fvars])

    def Δτ_stats_as_list(self, Δτ_stats):
        Δτ_stats_list = []
        for v in Δτ_stats.values():
            Δτ_stats_list.append(v['all'])               
            for vk, vv in v.items():
                if isinstance(vv, dict):    
                    Δτ_stats_list.append(vv['all'])

        return Δτ_stats_list
    