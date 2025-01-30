import gc
import math

import numpy as np

from pyfr.integrators.std.base import BaseStdIntegrator
from pyfr.mpiutil import get_comm_rank_root, mpi


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

        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:
            # Decide on the time step
            self.adjust_dt(t)

            # Take the step
            idxcurr = self.step(self.tcurr, self.dt)

            # We are not adaptive, so accept every step
            self._accept_step(self.dt, idxcurr)

            # Collect and store data in operable format
            if self.cfg.getbool('mesh', 'collect-statistics', False):

                if not (self.tcurr in self.tlist):
                    self.optimiser.collect_data()
                else:
                    comm, rank, root = get_comm_rank_root()
                    self.optimiser.process_statistics()
                    weights = comm.allgather(self.optimiser.median_weight)

#                    if not self.cfg.getbool('mesh', 'enable-relocator', False):
#                        self.optimiser.windows +=1
#
#                    elif self.optimiser.latest_performance > self.optimiser.best_performance:
#
#                        # Update the best performance
#                        self.optimiser.best_performance = self.optimiser.latest_performance
#                        self.optimiser.best_weight = self.optimiser.latest_weight
#
#                        # Continue to load-balance with new weights
#                        weights = comm.allgather(self.optimiser.median_weight)
#                    else:
#                        # Revert to best_performance weights with a larger window
#                        #   elements distribution different from last time, 
#                        #       so we can check performance replicability
#                        weights = comm.allgather(self.optimiser.best_weight)
#                        self.optimiser.windows +=1

                    # Decide on whether the latest weights worked better than the weights we get right now.
                    # This means from the mean other-time we got ...
                    #   remove out plugin time hidden in accept-steps
                    #   remove out time spent load balancing too.
                    #   remove out time spent in reinitialisation
                    #   remove out time spent in collecting data

                    if self.cfg.getbool('mesh', 'enable-relocator', False) and \
                       self.cfg.getbool('mesh', 'relocate-compute', False):
                        comm.barrier()

                        # Calculate target elements based on weights
                        t_nelems = self.load_relocator.firstguess_target_nelems(
                                                                            weights)

                        mesh, iters = self.load_relocator.diffuse_computation(
                                                                'compute', t_nelems)
                        gc.collect()
                        comm.barrier()

                        # if nelems_diff is list(zeros), don't do anything
                        new_ranks = self.load_relocator.new_ranks
                        if iters > 0:
                            comm.barrier()
                            # Reinitialize the system with the new mesh and solution
                            soln = self.load_relocator.reloc(
                                'compute', 'compute_new',
                                {m:s for m,s in zip(self.load_relocator.mm.etypes, 
                                                    self.soln)}, edim=2).values()

                            if len(new_ranks) == comm.size:
                                ranks_map = {i:i for i in range(comm.size)}
                            else:
                                comm, ranks_map = mpi.update_comm(new_ranks)

                            self.load_relocator.mm.update_mmesh_comm('compute_new',
                                                                comm, ranks_map)

                            # Reinitialize backend with the new communicator
                            self.backend()
                            self.system = self._systemcls(self.backend, mesh, 
                                                        list(soln), 
                                                        nregs=self.nregs, 
                                                        cfg=self.cfg)
                            self.system.commit()
                            self.system.preproc(self.tcurr, self._idxcurr)
                            # Delete all memoized cache attributes
                            for attr in dir(self):
                                if attr.startswith('_memoize_cache@'):
                                    delattr(self, attr) 
                            gc.collect()
                            comm.barrier()

                if (self.tcurr in self.tlist):
                    self.optimiser.empty_stats()


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