import numpy as np

from pyfr.optimisers.base import BaseLocalOptimiser


class BinaryStepper(BaseLocalOptimiser):
    name = 'binarystepper'
    systems = ['*']
    formulations = ['std', 'dual']

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        self.cost = intg.costs[self.cfg.get(cfgsect, 'cost')]        
        self.get_param = intg.get_parameters[self.cfg.get(cfgsect, 'parameter')]        
        self.set_param = intg.set_parameters[self.cfg.get(cfgsect, 'parameter')]

        self.prev_cost = 0
                
    def __call__(self, intg):

        if self.cost()[0] is None:
            return
        
        if self.prev_cost > self.cost()[0]:
            self.set_param(intg, self.get_param(intg) - 0.1)
        else:
            self.set_param(intg, self.get_param(intg) + 0.1)

        self.prev_cost = self.cost()[0]
            
        print('BinaryStepper: ', self.cost()[0], self.get_param(intg))
