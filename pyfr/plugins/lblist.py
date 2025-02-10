from pyfr.plugins.base import BaseSolnPlugin

class LblistPlugin(BaseSolnPlugin):
    name = 'lblist'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Output time step and last output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_last = intg.tcurr

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dt_out)

    def __call__(self, intg):
        if intg.tcurr - self.tout_last < self.dt_out - self.tol:
            return

        # Update the last output time
        self.tout_last = intg.tcurr
