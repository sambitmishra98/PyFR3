from time import perf_counter

from pyfr.integrators.dual.phys.base import BaseDualIntegrator


class BaseDualController(BaseDualIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Solution filtering frequency
        self._fnsteps = self.cfg.getint('soln-filter', 'nsteps', '0')

        # Fire off any event handlers if not restarting
        if not self.isrestart:
            self._run_plugins()
            self._run_optimisers()

    def _accept_step(self, idxcurr):
        self.tcurr += self._dt
        self.nacptsteps += 1
        self.nacptchain += 1

        # Filter
        if self._fnsteps and self.nacptsteps % self._fnsteps == 0:
            self.pseudointegrator.system.filt(idxcurr)

        self._invalidate_caches()

        # Run any plugins
        self._run_plugins()
        self._run_optimisers()

        # Clear the pseudo step info
        self.pseudointegrator.pseudostepinfo = []
        self.pseudointegrator.pseudostep_multipinfo = []

class DualNoneController(BaseDualController):
    controller_name = 'none'
    controller_has_variable_dt = False

    def advance_to(self, t):
        if t < self.tcurr:
            raise ValueError('Advance time is in the past')

        while self.tcurr < t:
            # Take the physical step

            # If self.perf_counter_info exists, then we need to collect perf.
            if self.performanceinfo is not None:
                tstart = perf_counter()
                self.step(self.tcurr, self._dt)
                self.performanceinfo = perf_counter() - tstart
            else: 
                self.step(self.tcurr, self._dt)
    
            # We are not adaptive, so accept every step
            self._accept_step(self.pseudointegrator._idxcurr)
