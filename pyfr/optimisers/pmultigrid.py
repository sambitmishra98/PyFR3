import numpy as np

from pyfr.optimisers.base import Parameter


class PMultigrid(Parameter):
    name = 'pmultigrid'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        # How to encode cycle. 0 means do no encoding.
        self.encoding = self.cfg.getint(cfgsect, 'encoding', 0)

        intg.get_parameters[self.name] = self.get_parameter
        intg.set_parameters[self.name] = self.set_parameter

    def __call__(self, intg):
        pass
        
    def get_parameter(self, intg, encoding = 1, iteration = -1):
        if encoding == 1:
            return intg.pseudointegrator.cstepsf_list[0][-1]
        
    def set_parameter(self, intg, value, encoding = 1, iteration = -1):
        if encoding == 1:
            intg.pseudointegrator.cstepsf_list[0][-1] = value
