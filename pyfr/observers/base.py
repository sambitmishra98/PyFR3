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

        # Get the above from config
        stages = self.cfg.getbool(cfgsect, 'observe-all-stages', False)
        levels = self.cfg.getbool(cfgsect, 'observe-all-levels', False)
        pniters = self.cfg.getbool(cfgsect, 'observe-all-pseudoiters', False)

        self._stages = intg.pseudointegrator.pintg.stage_nregs
        self._levels = intg.pseudointegrator._order + 1
        self._pniters = intg.pseudointegrator._maxniters

        # Make the above in one line
        self.cost_name = self.name + '-' + suffix if suffix else self.name

        # Get plot name
        self.plot_name = self.cfg.get(cfgsect, 'plot-name', self.cost_name)

        default_shape = (1, 1)

        # Initialise storage
        intg.costs[self.cost_name] = np.zeros(
            (self._stages, self._levels, self._pniters, *default_shape))

    def plot_intg_cost(self, plottable, name, if_log=True):
        import matplotlib.pyplot as plt
        import mpld3
        import scienceplots

        plt.style.use(['science', 'grid'])

        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', 'v', '^', 'D']

        # Extract the shape
        stages, levels, pseudo_iters, *rest = plottable.shape

        # Set zeros to nan
        plottable[plottable == 0] = np.nan

        # Create subplots for each level
        fig, axes = plt.subplots(levels, 1, figsize=(15, 5 * levels), sharex=True, sharey=True)

        # Iterate over the levels and plot the data for each id in the rest
        for l, ax in enumerate(axes):
            for id_comb in np.ndindex(*rest):
                label = f"id {id_comb}"

                if if_log:
                    ax.semilogy(plottable[:, l, :, *id_comb].flatten(),
                                label=label, linestyle=line_styles[id_comb[0] % len(line_styles)],
                                marker=markers[id_comb[1] % len(markers)])
                else:
                    ax.plot(plottable[:, l, :, *id_comb].flatten(),
                            label=label, linestyle=line_styles[id_comb[0] % len(line_styles)],
                            marker=markers[id_comb[1] % len(markers)])

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

        # Get the above from config
        stages = self.cfg.getbool(cfgsect, 'optimise-all-stages', False)
        levels = self.cfg.getbool(cfgsect, 'optimise-all-levels', False)
        piters = self.cfg.getbool(cfgsect, 'optimise-all-pseudoiters', False)

        self._stages = intg.pseudointegrator.pintg.stage_nregs if stages else 1
        self._levels = intg.pseudointegrator._order + 1 if levels else 1
        self._pniters = intg.pseudointegrator._maxniters if piters else 1
