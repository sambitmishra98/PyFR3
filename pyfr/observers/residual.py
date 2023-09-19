import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.observers import BaseCost

class SStResidualNorm(BaseCost):
    """
        # Residual in comparison with the steady-state solution
        # This is only applicable for implicit time integrators
        
    """
    name = 'res_l2'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self.comm, self.rank, self.root = get_comm_rank_root()

        self.cost_name = self.name + '-' + suffix

        # Initialise storage
        intg.costs[self.cost_name] = np.zeros(
            (self._stages, self._levels, self._pniters))

    def __call__(self, intg):

        self.plot_intg_cost(intg.costs['res_l2-p'], name = 'res_l2-p')

        # TODO
        # Create a place-holder for the compute time for stage, level, pseudo-iteration
        # Create an iff condition that collects the sum across the step function in the pseudointegrator.
        # Remember, we ultimately need to decrease something like 
        #   (Δ𝓡/nΔτ)/Δ𝓣
        # where 
        #       Δ𝓡 is some norm of residual, 
        #       n  is some norm of the number of pseudo-iterations, and
        #       Δ𝓣 is the compute time
        
        # We will also need the ratio Δ𝓡/Δτ field for knowing which cycle is suitable where.

    def plot_intg_cost(self, plottable, name):
        import matplotlib.pyplot as plt
        
        # Plot whatever is in the intg.costs['res_l2-p'] field
        # This is a 3D array of shape (stages, levels, pseudo-iterations)
        # For each of the 4 levels, we need a line in the plot
        # Each line in the plot is a concatenation of pseudo-iterations in each of the three stages

        # We need to make it more generic, so that it can plot any field in intg.costs
        stages = plottable.shape[0]
        levels = plottable.shape[1]
        pseudo_iters = plottable.shape[2]
        
        fig, axes = plt.subplots(1, 3, figsize=(15,5), sharey=True)

        # Set any zeros to blanks
        plottable[plottable == 0] = np.nan

        for i, ax in enumerate(axes):
            for l in range(levels):
                ax.semilogy(plottable[i,l,:], label=f'level {l}')
            ax.set_title(f"Stage {i}")
            ax.legend()

        # Concatenate across stages. Maintain the blanks
        
        fig, ax = plt.subplots(1, 1, figsize=(15,5), sharey=True)
        for l in range(levels):
            ax.semilogy(plottable[:,l,:].flatten(), label=f'level {l}')

        for s in range(1, stages):
            ax.axvline(pseudo_iters*s, color='k', linestyle='--')

        # Add grid
        ax.grid(which='both', axis='both', linestyle='--')

        # Set xlimits to the number of pseudo-iterations
        ax.set_xlim(0, pseudo_iters*stages)

        # Label the stages 
        for s in range(stages):
            ax.text(pseudo_iters*(s + 0.5), 1e-1, f'Stage {s}', fontsize=12)

        # xlabel and ylabel
        ax.set_xlabel('Pseudo-iteration')
        ax.set_ylabel(name)

        ax.set_title(f"Stages concatenated")
        # legend outside the plot
        ax.legend(bbox_to_anchor=(1, 0.5), loc='center left', borderaxespad=0.)
            
        fig.savefig(f"{name}.png", dpi=300)
        plt.close(fig)
