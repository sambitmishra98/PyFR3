from pyfr.mpiutil import get_comm_rank_root, mpi
import numpy as np

def init_csv(cfg, cfgsect, header, *, filekey='file', headerkey='header'):
    # Determine the file path
    fname = cfg.get(cfgsect, filekey)

    # Append the '.csv' extension
    if not fname.endswith('.csv'):
        fname += '.csv'

    # Open for appending
    outf = open(fname, 'a')

    # Output a header if required
    if outf.tell() == 0 and cfg.getbool(cfgsect, headerkey, True):
        print(header, file=outf)

    # Return the file
    return outf

class BaseObserver:
    name = None
    systems = None
    formulations = None
    config_name = None

    def __init__(self, intg, cfgsect, suffix=None):
        self.cfg = intg.cfg
        self.cfgsect = cfgsect

        self.suffix = suffix

        self.ndims = intg.system.ndims
        self.nvars = intg.system.nvars


class BaseCost(BaseObserver):
    prefix = 'cost'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        self.comm, self.rank, self.root = get_comm_rank_root()

        # Get the above from config
        stages = self.cfg.getbool(cfgsect, 'observe-all-stages', False)
        levels = self.cfg.getbool(cfgsect, 'observe-all-levels', False)
        pniters = self.cfg.getbool(cfgsect, 'observe-all-pseudoiters', False)

        self._stages = intg.pseudointegrator.pintg.stage_nregs
        self._levels = intg.pseudointegrator._order + 1
        self._pniters = intg.pseudointegrator._maxniters

        # Make the above in one line
        self.cost_name = self.name + '-' + suffix if suffix else self.name

        self.if_plot = self.cfg.getbool(cfgsect, 'if-plot', False)
        if self.if_plot:
            # Get plot name
            self.plot_name = self.cfg.get(cfgsect, 'plot-name', self.cost_name)


        self.if_write = self.cfg.getbool(cfgsect, 'if-write', False)
        if self.if_write:
            # Get file name
            self.file_name = self.cfg.get(cfgsect, 'file-name', self.cost_name)

            # Open the file
            if self.rank == self.root:
                self.outf = init_csv(self.cfg, cfgsect, self.cost_name)
            else:
                self.outf = None

        default_shape = (1,)

        # If self.name has `modes` in it, then we need to add the number of modes
        shape = (2*self._levels-1,) if 'modes' in self.name else default_shape

        # Initialise storage
        intg.costs[self.cost_name] = np.zeros(
            (self._stages, self._levels, self._pniters, *shape))

    def __call__(self, intg):
        if self.if_plot:
            self.plot_intg_cost(intg.costs[self.cost_name], 
                                name = self.plot_name, 
                                if_log = False)

    def plot_intg_cost(self, plottable, name, if_log=True):
        import matplotlib.pyplot as plt
        import mpld3
        import scienceplots

        plt.style.use(['science', 'grid'])

        # 7 colours for 7 modes
        line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
        markers     = ['o', 's' , 'v' , '^', 'D', 'P' , 'X' ]

        # Extract the shape
        stages, levels, pseudo_iters, *rest = plottable.shape

        # Set zeros to nan
        plottable[plottable == 0] = np.nan

        # Create subplots for each level
        fig, axes = plt.subplots(levels, 1, figsize=(15, 5 * levels), sharex=True, sharey=True)

        # Iterate over the levels and plot the data for each id in the rest
        for l, ax in enumerate(axes):
            for id_comb in range(rest[0]):
                label = f"id {id_comb}"

                if if_log: 
                    ax.semilogy(plottable[:, l, :, id_comb].flatten(), 
                                label     = label, 
                                linestyle = line_styles[id_comb],
                                marker    = markers[id_comb])
                else:      
                    ax.plot(plottable[:, l, :, id_comb].flatten(), 
                            label     = label, 
                            linestyle = line_styles[id_comb],
                            marker    = markers[id_comb])

                # Highlight the boundaries between pseudo-iterations
                for s in range(1, stages):
                    ax.axvline(pseudo_iters * s, color='k', linestyle='--')

                # Grid
                ax.grid(which='both', axis='both', linestyle='--')

                # Titles and labels
                ax.set_title(f"P-multigrid level {l}")
                ax.set_ylabel(name)

                # Legend outside the plot
                ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', borderaxespad=0.)

            # Common x label for the last subplot
            ax.set_xlabel('Pseudo-iteration')

        # Title for the entire figure
        fig.suptitle(f"Variation across all stages (concatenated) within a physical timestep")

        fig.tight_layout()
        fig.subplots_adjust(top=0.95)  # Allow some space for the suptitle

        # Save the figure
        fig.savefig(f"{name}.png", dpi=300)
        plt.close(fig)


class BaseParameter(BaseObserver):
    prefix = 'parameter'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Get bounds on the cost
        self.bounds = self.cfg.getliteral(cfgsect, 'bounds')

        self.parameter_name = self.name + '-' + suffix if suffix else self.name

        # We either need (stages, levels and pseudo_iters) or the values
        if self.cfg.hasopt(cfgsect, 'parameter-values'):
            parameter_values = self.cfg.getliteral(cfgsect, 'parameter-values')

            # Convert to a three-dimensional numpy array
            intg.parameters[self.parameter_name] = np.array(parameter_values, )
            
            self._stages, self._levels, self._pniters = intg.parameters[self.parameter_name].shape

        else:

            # Get the above from config
            stages = self.cfg.getbool(cfgsect, 'optimise-all-stages', False)
            levels = self.cfg.getbool(cfgsect, 'optimise-all-levels', False)
            piters = self.cfg.getbool(cfgsect, 'optimise-all-pseudoiters', False)

            self._stages = intg.pseudointegrator.pintg.stage_nregs if stages else 1
            self._levels = intg.pseudointegrator._order + 1 if levels else 1
            self._pniters = intg.pseudointegrator._maxniters if piters else 1

            if self.cfg.hasopt(cfgsect, 'variation'):
                if self.cfg.get(cfgsect, 'variation') == 'uniform':

                    uni = self.cfg.getfloat(cfgsect, 'variation-val')
                    intg.parameters[self.parameter_name] = uni * np.ones(
                        (self._stages, self._levels, self._pniters))
                    
                elif self.cfg.get(cfgsect, 'variation') == 'exp':
                    a = self.cfg.getfloat(cfgsect, 'variation-exp-a') # Starting value
                    b = self.cfg.getfloat(cfgsect, 'variation-exp-b') # Asymptoting value
                    c = self.cfg.getfloat(cfgsect, 'variation-exp-c') # Rate of change
                    
                    # Formula is y = a + (b-a) * (1-exp(-c*x))
                    y = lambda x: a + (b-a) * (1-np.exp(-c*x))
                    
                    intg.parameters[self.parameter_name] = np.ones(
                        (self._stages, self._levels, self._pniters))
                    
                    for s in range(self._stages):
                        for l in range(self._levels):
                            for p in range(self._pniters):
                                intg.parameters[self.parameter_name][s, l, p] = y(p)