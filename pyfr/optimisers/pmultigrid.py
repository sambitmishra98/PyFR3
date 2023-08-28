import numpy as np

from pyfr.optimisers.base import Modifier


class PMultigrid(Modifier):
    name = 'pmultigrid'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        self.modifierlist = np.zeros((1, 1))

    def __call__(self, intg):
        # Get the cycle from dual physical controller
        print(intg.pseudointegrator.cstepsf_list)
        
        # Modify the cycle, inrease the cycle by 1
        intg.pseudointegrator.cstepsf_list[0][-1] += 1        
        