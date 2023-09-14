import numpy as np

from pyfr.observers import BaseParameter

class PSmoothing(BaseParameter):
    name = 'psmoothing'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)
    
        _, cstepsf = zip(*self.cfg.getliteral(
            'solver-dual-time-integrator-multip', 'cycle'))

        print(f'cstepsf = {cstepsf}')

        def_csteps = cstepsf[int(suffix)]
        intg.parameters[self.name +'-'+ suffix] = (def_csteps * np.ones(
                (
                 self._stages, 
                 self._levels,
                 self._pniters, 
                 )))

        # Let us instead have the numbers from 1 to largest, 
        # so that we can monitor indices
        #intg.parameters[self.name] = np.arange(
        #    1.0, 1.0 + stages*levels*pseudo_iterations).reshape(
        #        (stages, 
        #         levels,
        #         pseudo_iterations, 
        #        ))

    def __call__(self, intg):
        print(f'from parameters: c = ',intg.parameters)
