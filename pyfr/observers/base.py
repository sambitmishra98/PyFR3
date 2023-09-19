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

        # Initialise storage
        intg.costs[self.cost_name] = np.zeros(
            (self._stages, self._levels, self._pniters))

    def plot_intg_cost(self, plottable, name, if_log=True):
        import matplotlib.pyplot as plt
        import mpld3
        
        # Create plots as amazingly as possible, as if towards a journal publication
        import scienceplots
        plt.style.use(['science', 'grid'])
        
        # I want to also see overlapping plots, so choose the line qualities accordingly
        # Make a list of line styles and markers, up to 5
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', 's', 'v', '^', 'D']
        
        
        # Expected: a 3D array of shape (stages, levels, pseudo-iterations)
        stages, levels, pseudo_iters = plottable.shape
        
        # Set any zeros to blanks
        plottable[plottable == 0] = np.nan

        fig, ax = plt.subplots(1, 1, figsize=(15,5))
        for l in range(levels):
            if if_log:
                ax.semilogy(plottable[:,l,:].flatten(), label=f'level {l}', 
                            linestyle=line_styles[l], marker=markers[l])
            else:
                ax.plot(plottable[:,l,:].flatten(), label=f'level {l}', 
                        linestyle=line_styles[l], marker=markers[l])

        for s in range(1, stages):
            ax.axvline(pseudo_iters*s, color='k', linestyle='--')

        # Add grid
        ax.grid(which='both', axis='both', linestyle='--')

        # Set xlimits to the number of pseudo-iterations
        ax.set_xlim(0, pseudo_iters*stages)

        # xlabel and ylabel
        ax.set_xlabel('Pseudo-iteration')
        ax.set_ylabel(name)

        ax.set_title(f"Variation within a physical timestep across all stages")
        # legend outside the plot
        ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left', borderaxespad=0.)
            
        fig.savefig(f"{name}.png", dpi=300)
#        mpld3.save_html(fig, f"{name}.html")
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
