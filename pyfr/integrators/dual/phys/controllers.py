from pyfr.integrators.dual.phys.base import BaseDualIntegrator


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

    def _reject_step(self, dt, idxold):
        if dt <= self.dtmin:
            raise RuntimeError('Minimum sized time step rejected')

        self.nacptchain = 0
        self.nrjctsteps += 1

        self._idxcurr = idxold

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

            # We are will soon be adaptive, so do not blindly accept every step
            self._accept_step(self._dt, idxcurr)
