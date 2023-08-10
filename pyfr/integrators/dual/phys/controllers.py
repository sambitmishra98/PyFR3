import math
from time import perf_counter

import numpy as np

from pyfr.integrators.dual.phys.base import BaseDualIntegrator
from pyfr.mpiutil import get_comm_rank_root, mpi


class BaseDualController(BaseDualIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Solution filtering frequency
        self._fnsteps = self.cfg.getint('soln-filter', 'nsteps', '0')

        # Fire off any event handlers if not restarting
        if not self.isrestart:
            self._run_plugins()

    def _accept_step(self, dt, idxcurr, err = None):
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

    def _reject_step(self, dt, idxold, err = None):

        if dt <= 1e-12:
            raise RuntimeError('Minimum sized time step rejected')

        self.nacptchain = 0
        self.nrjctsteps += 1

        self.pseudointegrator._idxcurr = idxold


class DualNoneController(BaseDualController):
    controller_name = 'none'
    controller_has_variable_dt = False

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:

            # Decide on the time step
            self.adjust_step(t)

            # Decide on the pseudo time step
            self.pseudointegrator.adjust_pseudo_step(self._dt)

            # Take the physical step
            idxcurr = self.step(self.tcurr, self._dt)

            # We are not adaptive, so accept every step
            self._accept_step(self._dt, idxcurr)


class DualPIController(BaseDualController):
    controller_name = 'pi'
    controller_has_variable_dt = True

    _atol = 1
    _rtol = 1

    _norm = 'l2'    
    _errprev = 1.0

    flag = 0         
    cost = 0
    the_factor = 1.01
        
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

        while self.tcurr < t:

            # Decide on the time step
            self.adjust_step(t)

            # Decide on the pseudo time step
            self.pseudointegrator.adjust_pseudo_step(self._dt)

            # Take the physical step
            start = perf_counter()
            idxcurr, idxprev, idxerr = self.step(self.tcurr, self._dt)
            cost = (perf_counter() - start) / self._dt
            
            did_cost_decrease = cost < self.the_factor*self.cost
            
            # Estimate the error
            err = self._errest(idxcurr, idxprev, idxerr)

            if did_cost_decrease and err < 10*self._errprev:
                fac = self.the_factor
            else:
                fac = 1/self.the_factor

            if err < 10*self._errprev:
                self._accept_step(self._dt, idxcurr)
            else:
                self.the_factor*=0.1
                self._reject_step(self._dt, idxprev)
                print(f"REJECTED STEP: dt = {self._dt:.5f},\t ")

            # Print with rounded to 3 decimal places
            print(
                  f"t = {self.tcurr:.5f},\t ",
                  f" dt = {self._dt:.5f},\t ",
                  f" cost = {cost:.5f},\t ",
                  f" self.cost = {self.cost:.5f},\t ",
                  f" err = {err:.5f},\t ",
                  )
            self.cost = cost            

            # Skip the first time we are asked to change the time step
            self._dt_in = fac*self._dt

            self._errprev = err
