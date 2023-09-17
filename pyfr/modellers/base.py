# Idea

# Given what we observe and what the cost is, we make our model.
# Types of models over which we can optimise:
# 0. Linear Regression
# 1. Gaussian Process Model (Followed by Bayesian Optimisation)
# 2. RNN (Optimising the weights of the RNN)
# 3. LSTM (Optimising the weights of the LSTM)
# 4. CNN (Optimising the weights of the CNN)

# This class will serve as the base for making model, 
# over which further optimisation is done.
class BaseModeller:
    name = None
    systems = None
    formulations = None

    def __init__(self, intg, cfgsect, suffix=None):
        self.cfg = intg.cfg
        self.cfgsect = cfgsect

        self.suffix = suffix


class BaseGlobalModeller(BaseModeller):
    prefix = 'global'
    
    
class BaseLocalModeller(BaseModeller):
    prefix = 'local'

    def __init__(self, intg, cfgsect, suffix=None):

        self._stages = intg.pseudointegrator.pintg.stage_nregs
        self._levels = intg.pseudointegrator._order + 1
        self._pniters = intg.pseudointegrator._maxniters
