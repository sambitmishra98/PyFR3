
from pyfr.optimisers.base import (BaseOptimiser, 
                                  BaseBayesianOptimiser, 
                                  BaseLocalOptimiser,
                                  Cost)
from pyfr.optimisers.runtime import RuntimeCost
from pyfr.util import subclass_where


def get_optimiser(prefix, name, *args, **kwargs):
    cls = subclass_where(BaseOptimiser, prefix=prefix, name=name)
    return cls(*args, **kwargs)
