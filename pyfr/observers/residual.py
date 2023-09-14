import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.observers import BaseCost

class SStResidualNorm(BaseCost):
    """
        # Residual in comparison with the steady-state solution
        # This is only applicable for implicit time integrators
        
    """
    name = 'sst_residual_norm'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self.comm, self.rank, self.root = get_comm_rank_root()

        # Write the above if conditions in a more compact way
        self.index = {'p': 0, 'u': 1, 'v': 2, 'w': 3}[suffix]

        self.cost_name = self.name +'-'+ suffix# + '-' + str(self.lp)

        # Initialise storage
        intg.costs[self.cost_name] = np.zeros(
            (self._stages, self._levels, self._pniters))

    def __call__(self, intg):

        print(f'from costs: {self.name} = ',intg.costs)

        self.postprocess(intg.costs[self.cost_name])
        
        # Clean intg.costs for the next time step as an array of zeros
        intg.costs = {self.cost_name: np.zeros(
            (self._stages, self._levels, self._pniters))}

    def postprocess(self, residual):
        # We have a 3D array of L₂ norm of residual for the pressure field
        # Axis 1 : stages
        # Axis 2 : levels
        # Axis 3 : pseudo_iterations
        # Example: from costs: sst_residual_norm =  {
            # 'sst_residual_norm-p': array([[[0.67901912, 0.67094941],
            #                                [1.22636501, 1.05798736]],
            #                               [[0.62648985, 0.56311111],
            #                                [0.83705973, 0.71545444]],
            #                               [[0.52693238, 0.49102683],
            #                                [0.64245283, 0.64679128]]])}

        # We want to compute ΔR/R = (Rₙ - Rₙ₋₁)/Rₙ₋₁, 
        # so we get (Rₙ - Rₙ₋₁)/Rₙ₋₁,  first along the pseudo_iterations axis
        rel = (residual[:, :, 1:] - residual[:, :, :-1]) / residual[:, :, :-1]

        print(f'from costs: {self.name} = ',rel)

        # We would get 
        #from costs: sst_residual_norm =  [[[-0.01188436]
        #                                   [-0.13729815]]
        #                                  [[-0.10116484]
        #                                   [-0.14527671]]
        #                                  [[-0.06814072]
        #                                   [ 0.00675295]]]

        rel[rel > 0] *= 1
        rel[rel <= 0] = 0

        # Store this as processed_cost, which is a 3D array of ΔR/R
        self.processed_cost = rel
        