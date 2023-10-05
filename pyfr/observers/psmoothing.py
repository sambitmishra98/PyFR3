import numpy as np

from pyfr.observers import BaseParameter


class PSmoothing(BaseParameter):
    name = 'psmoothing'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        if intg.parameters.get(self.parameter_name) is None:

            print(f"Not provided {self.parameter_name} in config file. "
                  f"Using default value.")

            _, cstepsf = zip(*self.cfg.getliteral(
                'solver-dual-time-integrator-multip', 'cycle'))

            def_csteps = cstepsf[int(suffix)]
            intg.parameters[self.parameter_name] = def_csteps * np.ones(
                    (self._stages, self._levels, self._pniters, ))

        # print(f"NEW: {intg.parameters[self.parameter_name]}")

    def __call__(self, intg):
        return
