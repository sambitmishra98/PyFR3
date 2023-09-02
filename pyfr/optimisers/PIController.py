import numpy as np

from pyfr.optimisers.base import BaseLocalOptimiser


class PIController(BaseLocalOptimiser):
    name = 'PIcontroller'
    systems = ['*']
    formulations = ['std', 'dual']

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        # You would have already given `cost` and `parameter` options in the config file.
        # They will be in the format cost1, cost2, cost3, etc. and param1, param2, param3, etc. respectively
        # 
        # self.costs is a dictionary with keys cost1, cost2, cost3, etc. and corresponding values as one float number each
        # self.parameters is a dictionary with keys param1, param2, param3, etc. and corresponding pointer to the locations in the solver itself.
        
        # We want this PI controller to control the parameter in accordance with the cost.

        # For now, let us assume that cost is runtime, and parameter is pmultigrid-0


        self._atol = self.cfg.getfloat(cfgsect, 'atol')
        self._rtol = self.cfg.getfloat(cfgsect, 'rtol')
        self._saff = self.cfg.getfloat(cfgsect, 'safety-factor', 0.8)

        self._alpha = self.cfg.getfloat(cfgsect, 'pi-alpha', 0.58)
        self._beta = self.cfg.getfloat(cfgsect, 'pi-beta', 0.42)

        self._minf = self.cfg.getfloat(cfgsect, 'min-fact', 0.9)
        self._maxf = self.cfg.getfloat(cfgsect, 'max-fact', 1.1)

        self._minp = self.cfg.getfloat(cfgsect, 'param-min')
        self._maxp = self.cfg.getfloat(cfgsect, 'param-max')

        self.dataset = {'parameters': [], 'costs': []}

        if self._minp > self._maxp:
            raise ValueError('Invalid param-min, param-max')
        
        if self._minf > 1 > self._maxf:
            raise ValueError('Invalid min-fact, max-fact')

    def __call__(self, intg):
        super().__call__(intg)

        if self.cost[0] is None:
            return        

        old_param = self.parameter
 
        if self.cost[0] > (self._atol+self._rtol*self.prev_cost[0]):
            aim = self._maxp
            self.PI_towards_aim(aim)

        self._post_call()
        print('ζ: ', intg.pseudointegrator.system.ac_zeta)

    def PI_towards_aim(self, aim):
        
        p_val = np.clip(aim, self._minp, self._maxp)
        f = (p_val - self.parameter)/self.parameter
        factor = np.clip(f*self._saff, self._minf, self._maxf)
        self.parameter = self.parameter*factor

        