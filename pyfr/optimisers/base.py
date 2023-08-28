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


class BaseOptimiser:
    
    name = None
    systems = None
    formulations = None
    
    def __init__(self, intg, cfgsect, suffix = None):
        self.cfg = intg.cfg
        self.cfgsect = cfgsect

        self.suffix = suffix

    def __call__(self, intg):
        pass


class Cost(BaseOptimiser):
    prefix = 'cost'    


class Parameter(BaseOptimiser):
    prefix = 'parameter'    


class BaseBayesianOptimiser(BaseOptimiser):
    prefix = 'bayes'
    
    
class BaseLocalOptimiser(BaseOptimiser):
    prefix = 'local'
