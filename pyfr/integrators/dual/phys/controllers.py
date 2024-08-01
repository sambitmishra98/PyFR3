from pyfr.integrators.dual.phys.base import BaseDualIntegrator


class BaseDualController(BaseDualIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Solution filtering frequency
        self._fnsteps = self.cfg.getint('soln-filter', 'nsteps', '0')

        # Fire off any event handlers if not restarting
        if not self.isrestart:
            self._run_plugins()

    def _accept_step(self, dt, idxcurr, err=None):
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
            self.adjust_dt(t)

            # Decide on the pseudo time step
            self.pseudointegrator.adjust_dtau(self.dt)

            # Take the physical step
            self.step(self.tcurr, self.dt)

            # We are not adaptive, so accept every step
            self._accept_step(self.dt, self.pseudointegrator._idxcurr)
