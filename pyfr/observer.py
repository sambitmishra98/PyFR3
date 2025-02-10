import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi

class Observer:
    def __init__(self):
        pass
    
    
class IntegratorObserver(Observer):
    def __init__(self):
        super().__init__()

    
class IntegratorPerformanceObserver(IntegratorObserver):

    def __init__(self, intg):
        super().__init__()

        self.intg = intg

        self.nskip = 1 + intg.cfg.getint('backend', 'collect-wait-times-len', 10000)
        self.windows = 1
        self.K_p = intg.cfg.getfloat('mesh', 'lb-proportionality', 1.0)

        # Collect the time difference between two steps
        self.nᵢ = 0  # Last number of function evaluations
        self.tᵢ = 0. # Last physical time
        self.pᵢ = 0. # Last plugin time

        self.stat_lists = {
            'Δt'    : [], 'Δn'    : [],
            'Δwait' : [], 'Δother': [], 'Δotherinverse': [],
#            'Δp'    : [], 
#            'Δlb'   : [], 'Δreset': [],
        }

        self.status = {'tcurr': 0, 'nfevals': 0}

        self.stats = {
            # Basics extracted
            'wait': (0,0,0), 'other': (0,0,0), 'total': (0,0,0), 
            #'plugin' : (0,0,0), 

            # Derivatives of the basics
            # 'compute': (0,0,0), 
            'otherinverse': (0,0,0), 
            'others-perdof': (0,0,0), 
            
            # Others
            'step-fevals': (0,0,0), 
            'weight': (0,0,0),
            
        }

        # Initialise csv
        comm, rank, root = get_comm_rank_root()
        self.csv = f'perf-_{rank}.csv'
        with open(self.csv, 'w') as f:
            row = []
            row.extend([k for k in self.status.keys()])
            row.extend([f'{k}-mean,{k}-std,{k}-med' for k in self.stats.keys()])
            f.write(','.join(row) + '\n')

    def collect_data(self):
        """
            Collect all statistics per unit physical time step.
        """

        n = self.intg._stepper_nfevals
        if n == self.nᵢ:
            return

        # Number of function evaluations difference    
        self.stat_lists['Δn'].append(n - self.nᵢ)
        self.nᵢ = n

        # Keep tab on status
        self.status['tcurr'] = self.intg.tcurr
        self.status['nfevals'] = n

        # Physical time difference
        self.stat_lists['Δt'].append(self.intg.tcurr - self.tᵢ)
        self.tᵢ = self.intg.tcurr

        # Get instantaneous wait times and sum over groups
        inst_rhs_wtimes = self.intg.system.instantaneous_rhs_wait_times()
        total_inst_rhs_wtime = inst_rhs_wtimes / self.stat_lists['Δn'][-1]
        self.stat_lists['Δwait'].append(total_inst_rhs_wtime)

        # Get instantaneous other times and sum over groups
        inst_rhs_otimes = self.intg.system.instantaneous_rhs_other_times()
        total_inst_rhs_otime = inst_rhs_otimes / self.stat_lists['Δn'][-1]
        self.stat_lists['Δother'       ].append(    total_inst_rhs_otime)
        self.stat_lists['Δotherinverse'].append(1 / total_inst_rhs_otime)

        # Plugin time difference
        #p_now, p_prev = self.intg._plugin_wtimes['common', None], self.pᵢ
        #self.stat_lists['Δp'].append(p_now - p_prev)
        #self.pᵢ = p_now

        #self.stat_lists['Δp'].append(self.intg.plugin_diff)

        # Load balancing time difference, if exists
#        if hasattr(self.intg, 'lb_diff'):
#            self.stat_lists['Δlb'].append(self.intg.lb_diff)

        # Re-initialisation time difference, if load-blancing is enabled
#        if hasattr(self.intg, 'reset_diff'):
#            self.stat_lists['Δreset'].append(self.intg.reset_diff)

    def process_statistics(self):

        comm, rank, root = get_comm_rank_root('compute')

        self.stats |= self.rhs_times_from_self()
        self.stats |= self.others()

        # Write to csv
        # ---------------------------------------------------------------------
        with open(self.csv, 'a') as f:
            row = []
            row.extend([self.status['tcurr'], self.status['nfevals'],])
            row.extend([x for y in self.stats.values() for x in y])
            f.write(','.join(map(str, row)) + '\n')

        # Print on a separate csv, wait and other and plugin times
        with open(f'./waits/wait-{rank}.csv', 'w') as f:
            # All list in one column
            for i in range(len(self.stat_lists['Δwait'])):
                f.write(f'{self.stat_lists["Δwait"][i]}\n')
                
        with open(f'./others/other-{rank}.csv', 'w') as f:
            # All list in one column
            for i in range(len(self.stat_lists['Δother'])):
                f.write(f'{self.stat_lists["Δother"][i]}\n')

        # Write value of solver.tcurr to csv
        with open(f'./weights/weight-{rank}.csv', 'a') as f:
            # Just print weight
            f.write(f'{int(self.stats["weight"][2])}\n')

        # Write value of solver.tcurr to csv
        with open(f'./statistics/statistics-{rank}.csv', 'a') as f:
            row = [ str(self.Nₑ)]
            row.extend([str(self.otherinverse)])
            row.extend([str(int(x)) for x in self.gathered_Δd])
            row.extend([str(x) for x in self.actual_times_overall])
            f.write(','.join(row) + '\n')
        # ---------------------------------------------------------------------


    def stats_from_list(self, data_list):
        # Always skip the first few steps 
        if len(data_list) < self.nskip:
            return (0,0,0)

        return (np.mean(  data_list[self.nskip:-1]), 
                np.std(   data_list[self.nskip:-1]), 
                np.median(data_list[self.nskip:-1]),
                )

    def empty_stats(self):

        for k in self.stat_lists.keys():
            self.stat_lists[k].clear()
            
        for k in self.stats.keys():
            self.stats[k] = (0,0,0)

    @property
    def Nₑ(self):
        return sum({etype: neles for etype, (nupts, nvars, neles) 
                            in self.intg.system.ele_shapes.items()}.values())

    @property
    def Nₑ_global(self):
        comm, rank, root = get_comm_rank_root()
        return comm.allreduce(self.Nₑ)

    def rhs_times_from_self(self):

        self.stat_lists['Δtotal'] = [
            wait+other for wait,other in zip(self.stat_lists['Δwait'], 
                                             self.stat_lists['Δother'])]

        temp_stats = {
            'wait' : self.stats_from_list(self.stat_lists['Δwait' ]),
            'other': self.stats_from_list(self.stat_lists['Δother']),
            'otherinverse': self.stats_from_list(self.stat_lists['Δotherinverse']),
            'total': self.stats_from_list(self.stat_lists['Δtotal']),
        }
        
        return temp_stats

    def weight_definition(self, stat_lists):
        comm, rank, root = get_comm_rank_root()
    
        w_local = [self.Nₑ * o for o in stat_lists['Δotherinverse']]
        w_all = np.array(comm.allgather(w_local))
        w_all /= w_all.sum(axis=0)
        w_all = np.round(w_all * self.Nₑ_global).astype(int)

        curr_dist = np.array(comm.allgather(self.Nₑ))

        curr_dist = curr_dist[:, None]
        curr_Nₑ = np.tile(curr_dist, (1, w_all.shape[1]))
        new_all = (curr_Nₑ + self.K_p * (w_all - curr_Nₑ)).astype(int)

        return new_all[rank].tolist()

    def others(self):

        self.stat_lists['weight'] = self.weight_definition(self.stat_lists)

        temp_stats = {
            'step-fevals': self.stats_from_list(self.stat_lists['Δn']),
            'weight'     : self.stats_from_list(self.stat_lists['weight']),
        }
        
        return temp_stats

    def allgather_stats_from_list(self, list_name):
        comm, rank, root = get_comm_rank_root()
        cpd = np.array(comm.allgather(self.stat_lists[list_name])).sum(axis=0)
        return cpd.mean(), cpd.std()

    @property
    def actual_times_overall(self):
        Δn_mean = self.stats_from_list(self.stat_lists['Δn'])[0]
        global_Δd = self.intg._gndofs * Δn_mean
        mean, std = self.allgather_stats_from_list('Δtotal')
        return mean/global_Δd, std/global_Δd

    @property
    def lost_time(self):
        #get mean and std
        return ((self.stats['wait'][0])/self.Δd, 
                (self.stats['wait'][1])/self.Δd,
                ) if self.Δd != 0 else (0,0)

    @property
    def otherinverse(self):
        return self.stats['otherinverse'][0]

    @property
    def median_weight(self):
        return self.stats['weight'][2].astype(float)

    @property
    def Δd(self):
        Δn_mean = self.stats_from_list(self.stat_lists['Δn'])[0]
        return sum(self.intg.system.ele_ndofs) * Δn_mean

    @property
    def gathered_Δd(self):
        comm, rank, root = get_comm_rank_root()
    
        return comm.allgather(self.Δd)

    def increase_capture_window(self):
        self.windows += 1