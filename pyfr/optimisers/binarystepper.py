from pyfr.optimisers.base import BaseLocalOptimiser

class BinaryStepper(BaseLocalOptimiser):
    name = 'binarystepper'
    systems = ['*']
    formulations = ['std', 'dual']

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        self.parameter_name = 'zeta'
        self.cost_name = 'sst_residual_norm-p'

        self.factor = self.cfg.getfloat(cfgsect, 'factor', 0.1)

    def __call__(self, intg):

        # Modify the intg.parameters dictionary with the new values
        intg.parameters[self.parameter_name] += intg.model*self.factor
