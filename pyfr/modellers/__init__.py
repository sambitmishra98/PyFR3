from pyfr.modellers.base import (BaseModeller, 
                                 BaseLocalModeller,
                                 BaseGlobalModeller, 
                                 )

from pyfr.modellers.binarymodeller import BinaryModeller

from pyfr.util import subclass_where

def get_modeller(prefix, name, *args, **kwargs):
    cls = subclass_where(BaseModeller, prefix=prefix, name=name)
    return cls(*args, **kwargs)

