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

    def process_cost(self, costlist, n_skip = 2, n_capture = 8):
        if len(costlist) < n_skip + n_capture:
            return None, None
        else:
            captured = costlist[n_skip:]
            cost = np.mean(captured)
            cost_err  = np.std(captured)/np.sqrt(len(captured))/cost

            return cost, cost_err


class Modifier(BaseOptimiser):
    prefix = 'modifier'    


class BaseBayesianOptimiser(BaseOptimiser):
    prefix = 'bayes'
    
    
class BaseLocalOptimiser(BaseOptimiser):
    prefix = 'local'
