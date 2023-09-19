from pyfr.observers.base import (BaseObserver, BaseCost, BaseParameter)

from pyfr.observers.zeta import Zeta
from pyfr.observers.psmoothing import PSmoothing

from pyfr.observers.residual import SStResidualNorm
from pyfr.observers.walltime import WallTime

from pyfr.util import subclass_where

def get_observer(prefix, name, *args, **kwargs):
    cls = subclass_where(BaseObserver, prefix=prefix, name=name)
    return cls(*args, **kwargs)

