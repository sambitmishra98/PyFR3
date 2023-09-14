from pprint import pprint

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
        stages = self.cfg.getbool(cfgsect, 
            'observe-all-stages', False)
        levels = self.cfg.getbool(cfgsect, 
            'observe-all-levels', False)
        pniters = self.cfg.getbool(cfgsect, 
            'observe-all-pseudoiterations', False)

        self._stages = intg.pseudointegrator.pintg.stage_nregs if stages else 1
        self._levels = intg.pseudointegrator._order + 1 if levels else 1
        self._pniters = intg.pseudointegrator._maxniters if pniters else 1


class BaseParameter(BaseObserver):
    prefix = 'parameter'

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Get bounds on the cost
        self.bounds = self.cfg.getliteral(cfgsect, 'bounds')

        # Get the above from config
        stages = self.cfg.getbool(cfgsect, 
            'optimise-all-stages', False)
        levels = self.cfg.getbool(cfgsect, 
            'optimise-all-levels', False)
        pniters = self.cfg.getbool(cfgsect, 
            'optimise-all-pseudoiterations', False)

        self._stages = intg.pseudointegrator.pintg.stage_nregs if stages else 1
        self._levels = intg.pseudointegrator._order + 1 if levels else 1
        self._pniters = intg.pseudointegrator._maxniters if pniters else 1
