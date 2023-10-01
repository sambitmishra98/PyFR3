from pyfr.solvers.acnavstokes.elements import ACNavierStokesElements
from pyfr.solvers.acnavstokes.inters import (ACNavierStokesBaseBCInters,
                                             ACNavierStokesIntInters,
                                             ACNavierStokesMPIInters)
from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionSystem


class ACNavierStokesSystem(BaseAdvectionDiffusionSystem):
    name = 'ac-navier-stokes'

    elementscls = ACNavierStokesElements
    intinterscls = ACNavierStokesIntInters
    mpiinterscls = ACNavierStokesMPIInters
    bbcinterscls = ACNavierStokesBaseBCInters

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ac_zeta = self.cfg.getfloat('solver', 'ac-zeta')

    def _prepare_kernels(self, t, uinbank, foutbank):
        _, binders = self._get_kernels(uinbank, foutbank)

        for b in self._bc_inters:
            b.prepare(t)

        for b in binders: 
            b(t=t, ac_zeta=self.ac_zeta)
