import numpy as np

from pyfr.optimisers.base import BaseLocalOptimiser


class BinaryStepper(BaseLocalOptimiser):
    name = 'binarystepper'
    systems = ['*']
    formulations = ['std', 'dual']

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

    def __call__(self, intg):
        super().__call__(intg)

        print(self.costs['runtime'][0])

        if self.costs['runtime'][0] is None:
            return
        
        pmg = self.parameters['pmultigrid']
        if self.prev_costs['runtime'][0] > self.costs['runtime'][0]:
            self.parameters = [pmg + 0.1,]
        else:
            self.parameters = [pmg - 0.1,]
            
        print(self.parameters['pmultigrid'])
        
        self._post_call()
