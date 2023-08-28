
from pyfr.optimisers.base import (BaseOptimiser, 
                                  BaseBayesianOptimiser, 
                                  BaseLocalOptimiser,
                                  Cost,
                                  Parameter)
from pyfr.optimisers.binarystepper import BinaryStepper
from pyfr.optimisers.runtime import RuntimeCost
from pyfr.optimisers.pmultigrid import PMultigrid
from pyfr.util import subclass_where

def get_optimiser(prefix, name, *args, **kwargs):
    cls = subclass_where(BaseOptimiser, prefix=prefix, name=name)
    return cls(*args, **kwargs)
