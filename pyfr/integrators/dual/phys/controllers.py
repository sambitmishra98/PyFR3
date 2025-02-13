import gc
from time import perf_counter
from pyfr.integrators.dual.phys.base import BaseDualIntegrator

from pyfr.backends import get_backend
from pyfr.mpiutil import append_comm_rank_root, get_comm_rank_root, mpi
from pyfr.integrators.dual.pseudo import get_pseudo_integrator
from pyfr.integrators.dual.pseudo.multip import DualMultiPIntegrator


class BaseDualController(BaseDualIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Solution filtering frequency
        self._fnsteps = self.cfg.getint('soln-filter', 'nsteps', '0')

        # Fire off any event handlers if not restarting
        if not self.isrestart:
            self._run_plugins()

    def _accept_step(self, dt, idxcurr):
        self.tcurr += dt
        self.nacptsteps += 1
        self.nacptchain += 1

        # Filter
        if self._fnsteps and self.nacptsteps % self._fnsteps == 0:
            self.pseudointegrator.system.filt(idxcurr)

        self._invalidate_caches()

        # Run any plugins
        self._run_plugins()

        # Clear the pseudo step info
        self.pseudointegrator.pseudostepinfo = []


class DualNoneController(BaseDualController):
    controller_name = 'none'
    controller_has_variable_dt = False

    lb_tend = perf_counter()

    def advance_to(self, t):
        base_comm, base_rank, base_root = get_comm_rank_root('base')
        compute_comm, compute_rank, compute_root = get_comm_rank_root('compute')

        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:
            # Decide on the time step
            self.adjust_dt(t)

            # Decide on the pseudo time step
            self.pseudointegrator.adjust_dtau(self.dt)

            if compute_comm != mpi.COMM_NULL:

                # Take the physical step
                self.step(self.tcurr, self.dt)

                # We are not adaptive, so accept every step
                self._accept_step(self.dt, self.pseudointegrator._idxcurr)

            if compute_comm == mpi.COMM_NULL:
                raise NotImplementedError('Not supported yet')

            # Collect and store data in operable format
            if self.cfg.getbool('mesh', 'collect-statistics', False):

                if not self.tcurr in self.tlist:
                    if compute_comm != mpi.COMM_NULL:
                        self.optimiser.collect_data()
                else:
                    base_comm.barrier()
                    if base_comm != mpi.COMM_NULL:
                        base_comm.barrier()
                        if base_rank == base_root:
                            lb_tstart = perf_counter()

                    if compute_comm != mpi.COMM_NULL:
                        self.optimiser.process_statistics()
                        weights = self.optimiser.median_weight
                        print('Median Weights:', weights)

                    weights = [w for w in base_comm.allgather(weights) if w is not None]

                    if self.cfg.getbool('mesh', 'enable-relocator', False) and self.cfg.getbool('mesh', 'relocate-compute', False):

                        # Calculate target elements based on weights
                        t_nelems = self.load_relocator.firstguess_target_nelems(weights)

                        soln_dict = {m:s for m,s in zip(self.load_relocator.mm.etypes, self.soln)}
 
                        print("TARGET NELEMS", t_nelems, flush=True)

                        mesh = self.load_relocator.diffuse_computation('compute', t_nelems)

                        print(f"{compute_rank} Diffusion completed", flush=True)
                        soln_dict = self.load_relocator.reloc('compute', 'compute_new', soln_dict, edim=2)

                        self.load_relocator.mm.move_mmesh('compute_new', 'compute')
                        self.load_relocator.mm.copy_mmesh('compute', 'compute_new')

                        compute_comm, compute_ranks_map = mpi.update_comm([i for i, w in enumerate(t_nelems) if w > 0])
                        self.load_relocator.mm.update_mmesh_comm('compute', compute_comm, compute_ranks_map)
                        self.load_relocator.mm.update_mmesh_comm('compute_new', compute_comm, compute_ranks_map)

                        if compute_comm == mpi.COMM_NULL: compute_rank, compute_root = None, None
                        else:                             compute_rank, compute_root = compute_comm.rank, 0
                        
                        append_comm_rank_root('compute', compute_comm, compute_rank, compute_root, None)

                        if compute_comm != mpi.COMM_NULL:
                            gc.collect()
                            
                            be_name = self.backend.name
                            self.backend = get_backend(be_name, self.cfg)
                            #self.backend()

                            # self.system = self._systemcls(self.backend, mesh, list(soln_dict.values()), nregs=self.nregs, cfg=self.cfg)
                            #self._reget_plugins()

                            # Get the pseudo-integrator
                            #self.pseudointegrator = get_pseudo_integrator(
                            #    self.backend, self.pseudointegrator.pintg._systemcls, mesh, list(soln_dict.values()), self.cfg, self.stepper_nregs,
                            #    self.stage_nregs, self.dt
                            #)
                            
                            systemcls = self.pseudointegrator.pintg._systemcls
                            
                            self.pseudointegrator = DualMultiPIntegrator(self.backend, systemcls, mesh, list(soln_dict.values()), self.cfg, self.stepper_nregs, 
                                                                         self.stage_nregs, self.dt)

                            # Event handlers for advance_to
                            self.plugins = self._reget_plugins()

                            # Commit the pseudo integrators now we have the plugins
                            self.pseudointegrator.commit()

                            #self.system.commit()
                            #self.system.preproc(self.tcurr, self._idxcurr)

                            # Delete all memoized cache attributes
                            for attr in dir(self):
                                if attr.startswith('_memoize_cache@'):
                                    delattr(self, attr) 
    
                        gc.collect()

                    if base_comm != mpi.COMM_NULL:
                        base_comm.barrier()

                        if base_rank == base_root:
                            lb_start_diff_end = lb_tstart - self.lb_tend
                            self.lb_tend = perf_counter()
                            lb_end_diff_start = self.lb_tend - lb_tstart
                            print(f"\n----------------\nWall-times for non-loadbalance and load-balance respectively = \n{lb_start_diff_end}\n{lb_end_diff_start}", flush=True)

                if self.tcurr in self.tlist:
                    self.optimiser.empty_stats()
