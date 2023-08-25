
from pyfr.optimisers.base import (BaseOptimiser, 
                                  BaseBayesianOptimiser, 
                                  BaseLocalOptimiser)
from pyfr.optimisers.printcost import PrintCostOptimiser
from pyfr.util import subclass_where


def get_optimiser(prefix, name, *args, **kwargs):
    cls = subclass_where(BaseOptimiser, prefix=prefix, name=name)
    return cls(*args, **kwargs)
