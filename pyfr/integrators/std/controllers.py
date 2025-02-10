import gc
import math

import numpy as np

from pyfr.integrators.std.base import BaseStdIntegrator
from pyfr.mpiutil import append_comm_rank_root, get_comm_rank_root, mpi

from pyfr.loadrelocator import LoadRelocator

class BaseStdController(BaseStdIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure the system is compatible with our formulation/controller
        self.system.elementscls.validate_formulation(self)

        # Solution filtering frequency
        self._fnsteps = self.cfg.getint('soln-filter', 'nsteps', '0')

        # Stats on the most recent step
        self.stepinfo = []

        # Fire off any event handlers if not restarting
        if not self.isrestart:
            self._run_plugins()

        self._current_perf = 0

    def _accept_step(self, dt, idxcurr, err=None):
        self.tcurr += dt
        self.nacptsteps += 1
        self.nacptchain += 1
        self.stepinfo.append((dt, 'accept', err))

        self._idxcurr = idxcurr

        # Filter
        if self._fnsteps and self.nacptsteps % self._fnsteps == 0:
            self.system.filt(idxcurr)

        self._invalidate_caches()

        # Run any plugins
        self._run_plugins()

        # Clear the step info
        self.stepinfo = []

    def _accept_step_empty_rank(self, dt, idxcurr, err=None):
        self.tcurr += dt
        self.nacptsteps += 1
        self.nacptchain += 1
        self.stepinfo.append((dt, 'accept', err))

        self._idxcurr = idxcurr

        # Run any plugins
        self._run_plugins()

        # Clear the step info
        self.stepinfo = []

    def _reject_step(self, dt, idxold, err=None):
        if dt <= self.dtmin:
            raise RuntimeError('Minimum sized time step rejected')

        self.nacptchain = 0
        self.nrjctsteps += 1
        self.stepinfo.append((dt, 'reject', err))

        self._idxcurr = idxold


class StdNoneController(BaseStdController):
    controller_name = 'none'
    controller_has_variable_dt = False

    @property
    def controller_needs_errest(self):
        return False

    def advance_to(self, t):
        base_comm, base_rank, base_root = get_comm_rank_root('base')
        compute_comm, compute_rank, compute_root = get_comm_rank_root('compute')

        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:
            # Decide on the time step
            self.adjust_dt(t)

            if compute_comm != mpi.COMM_NULL:

                # Take the step
                idxcurr = self.step(self.tcurr, self.dt)

                # We are not adaptive, so accept every step
                self._accept_step(self.dt, idxcurr)

            # Get idxcurr from root rank to rest
            idxcurr = base_comm.bcast(self._idxcurr, root=base_root)

            if compute_comm == mpi.COMM_NULL:
                self._accept_step_empty_rank(self.dt, idxcurr)

            base_comm.barrier()

            # Collect and store data in operable format
            if self.cfg.getbool('mesh', 'collect-statistics', False):

                if not self.tcurr in self.tlist:
                    if compute_comm != mpi.COMM_NULL:
                        self.optimiser.collect_data()
                else:

                    if compute_comm != mpi.COMM_NULL:
                        self.optimiser.process_statistics()
                        weights = self.optimiser.median_weight
                    else:
                        if self.optimiser.n_tlist_rank_add is not None:
                            if self.tcurr == self.tlist[self.optimiser.n_tlist_rank_add]:
                                weights = -1
                            else:
                                weights = None
                            
                    weights = [w for w in base_comm.allgather(weights) if w is not None]

                    # If weights is -1 anywhere, then reset it to the value left of it
                    if -1 in weights:
                        for i, w in enumerate(weights):
                            if w == -1:
                                weights[i] = weights[i-1]

                    if self.cfg.getbool('mesh', 'enable-relocator', False) and self.cfg.getbool('mesh', 'relocate-compute', False):

                        # REMOVE RANK OR NOT ?????
                        # HERE !!!!
                        if self.optimiser.n_tlist_rank_remove is not None:
                            if self.tcurr == self.tlist[self.optimiser.n_tlist_rank_remove]:
                                weights[-1] = 0

                        # Calculate target elements based on weights
                        t_nelems = self.load_relocator.firstguess_target_nelems(weights)

                        # If t_nelems length is 3 but compute_comm is null, then we need to add in a rank
                        if len(t_nelems) == 3:

                            if self.optimiser.n_tlist_rank_add is not None:
                                if compute_comm == mpi.COMM_NULL and self.tcurr == self.tlist[self.optimiser.n_tlist_rank_add]: 
                                    adding_rank = 1
                                else:
                                    adding_rank = 0
                            else:                                                               
                                adding_rank = 0

                            adding_rank = base_comm.allgather(adding_rank)

                        else:
                            adding_rank = [0, 0]


                        soln_dict = {m:s for m,s in zip(self.load_relocator.mm.etypes, self.soln)}
 
                        # If even one of them is 1, then we need to add in a rank
                        if 1 in adding_rank:
                            temp_compute_comm, temp_compute_ranks_map = mpi.update_comm([i for i, _ in enumerate(t_nelems)])
                            append_comm_rank_root('compute', temp_compute_comm, temp_compute_comm.rank, 0, None)

                            self.load_relocator.mm.update_mmesh_comm('compute'    , temp_compute_comm, temp_compute_ranks_map)
                            self.load_relocator.mm.update_mmesh_comm('compute_new', temp_compute_comm, temp_compute_ranks_map)
                            
                            temp_mesh, soln_dict = self.load_relocator.add_rank('compute_new', soln_dict, from_rank = 0, to_rank = 2)

                            self.load_relocator = LoadRelocator(temp_mesh)

                            mesh = self.load_relocator.mm.get_mmesh('compute').to_mesh()

                        if compute_comm != mpi.COMM_NULL or 1 in adding_rank:

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
                            self.backend()
                            self.system = self._systemcls(self.backend, mesh, list(soln_dict.values()), nregs=self.nregs, cfg=self.cfg)
                            self._reget_plugins()
                            self.system.commit()
                            self.system.preproc(self.tcurr, self._idxcurr)
                            # Delete all memoized cache attributes
                            for attr in dir(self):
                                if attr.startswith('_memoize_cache@'):
                                    delattr(self, attr) 
    
                        gc.collect()

                        if base_comm != mpi.COMM_NULL:
                            base_comm.barrier()

                if self.tcurr in self.tlist:
                    self.optimiser.empty_stats()
                    print(f"{compute_rank} is alive", flush=True)


class StdPIController(BaseStdController):
    controller_name = 'pi'
    controller_has_variable_dt = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sect = 'solver-time-integrator'

        # Error tolerances
        self._atol = self.cfg.getfloat(sect, 'atol')
        self._rtol = self.cfg.getfloat(sect, 'rtol')

        if self._atol < 10*self.backend.fpdtype_eps:
            raise ValueError('Absolute tolerance too small')

        if self._rtol < 10*self.backend.fpdtype_eps:
            raise ValueError('Relative tolerance too small')

        # Error norm
        self._norm = self.cfg.get(sect, 'errest-norm', 'l2')
        if self._norm not in {'l2', 'uniform'}:
            raise ValueError('Invalid error norm')

        # PI control values
        self._alpha = self.cfg.getfloat(sect, 'pi-alpha', 0.58)
        self._beta = self.cfg.getfloat(sect, 'pi-beta', 0.42)

        # Estimate of previous error
        self._errprev = 1.0

        # Step size adjustment factors
        self._saffac = self.cfg.getfloat(sect, 'safety-fact', 0.8)
        self._maxfac = self.cfg.getfloat(sect, 'max-fact', 2.5)
        self._minfac = self.cfg.getfloat(sect, 'min-fact', 0.3)

        if not self._minfac < 1 <= self._maxfac:
            raise ValueError('Invalid max-fact, min-fact')

    @property
    def controller_needs_errest(self):
        return True

    def _errest(self, rcurr, rprev, rerr):
        comm, rank, root = get_comm_rank_root()

        # Get a set of kernels to estimate the integration error
        ekerns = self._get_reduction_kerns(rcurr, rprev, rerr, method='errest',
                                           norm=self._norm)

        # Bind the dynamic arguments
        for kern in ekerns:
            kern.bind(self._atol, self._rtol)

        # Run the kernels
        self.backend.run_kernels(ekerns, wait=True)

        # Pseudo L2 norm
        if self._norm == 'l2':
            # Reduce locally (element types + field variables)
            err = np.array([sum(v for k in ekerns for v in k.retval)])

            # Reduce globally (MPI ranks)
            comm.Allreduce(mpi.IN_PLACE, err, op=mpi.SUM)

            # Normalise
            err = math.sqrt(float(err) / self._gndofs)
        # Uniform norm
        else:
            # Reduce locally (element types + field variables)
            err = np.array([max(v for k in ekerns for v in k.retval)])

            # Reduce globally (MPI ranks)
            comm.Allreduce(mpi.IN_PLACE, err, op=mpi.MAX)

            # Normalise
            err = math.sqrt(float(err))

        return err if not math.isnan(err) else 100

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        # Constants
        maxf = self._maxfac
        minf = self._minfac
        saff = self._saffac
        sord = self.stepper_order

        expa = self._alpha / sord
        expb = self._beta / sord

        while self.tcurr < t:
            # Adjust current time step per target t
            self.adjust_dt(t)

            self.dt = max(self.dt, self.dtmin)

            # Take the step
            idxcurr, idxprev, idxerr = self.step(self.tcurr, self.dt)

            # Estimate the error
            err = self._errest(idxcurr, idxprev, idxerr)

            # Decide if to accept or reject the step
            if err < 1.0:
                self._errprev = err
                self._accept_step(self.dt, idxcurr, err=err)
            else:
                self._reject_step(self.dt, idxprev, err=err)

            # Adjust time step per PI controller
            fac = err**-expa * self._errprev**expb
            fac = min(maxf, max(minf, saff*fac))

            # Compute the next time step
            self.dt_fallback = fac*self.dt