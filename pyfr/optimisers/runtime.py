import numpy as np

from pyfr.optimisers.base import Cost


class RuntimeCost(Cost):
    name = 'runtime'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        self.costlist = np.zeros((1, 1))

        intg.costs[self.name] = self.process_cost

    def __call__(self, intg):

        # Append performance info to the 1-D matrix
        self.costlist = np.append(self.costlist, intg.performanceinfo)

    def process_cost(self, n_skip = 2, n_capture = 8):
        if len(self.costlist) < n_skip + n_capture:
            return None, None
        else:
            captured = self.costlist[n_skip:]
            cost = np.mean(captured)
            cost_err  = np.std(captured)/np.sqrt(len(captured))/cost

            return cost, cost_err
