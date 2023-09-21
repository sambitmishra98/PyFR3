from collections import defaultdict, deque
from configparser import NoOptionError
import re

from pyfr.integrators.base import BaseCommon
from pyfr.plugins import get_plugin


class BaseDualPseudoIntegrator(BaseCommon):
    formulation = 'dual'
    aux_nregs = 0

    def __init__(self, backend, systemcls, rallocs, mesh,
                 initsoln, cfg, stepper_nregs, stage_nregs, dt):
        self.backend = backend
        self.rallocs = rallocs
        self.isrestart = initsoln is not None
        self.cfg = cfg
        self._dt = dt

        sect = 'solver-time-integrator'

        self._dtaumin = 1.0e-12
        self._dtau = cfg.getfloat(sect, 'pseudo-dt')

        self.maxniters = cfg.getint(sect, 'pseudo-niters-max', 0)
        self.minniters = cfg.getint(sect, 'pseudo-niters-min', 0)

        if self.maxniters < self.minniters:
            raise ValueError('The maximum number of pseudo-iterations must '
                             'be greater than or equal to the minimum')

        if (self.pseudo_controller_needs_lerrest and
            not self.pseudo_stepper_has_lerrest):
            raise TypeError('Incompatible pseudo-stepper/pseudo-controller '
                            'combination')

        # Amount of stage storage required by DIRK stepper
        self.stage_nregs = stage_nregs

        # Amount of temp storage required by physical stepper
        self.stepper_nregs = stepper_nregs

        # Amount of temp storage required to store the previous and current modes
        self.modes_nregs = (cfg.getint('solver', 'order') + 1)**2

        # Residual in pseudo-time
        pseudo_residual_nregs = 1

        source_nregs = 1

        # Determine the amount of temp storage required in total
        self.nregs = (self.pseudo_stepper_nregs + self.stepper_nregs +
                      self.stage_nregs + source_nregs + 
                      self.modes_nregs + pseudo_residual_nregs + 
                      self.aux_nregs)

        # Construct the relevant system
        self.system = systemcls(backend, rallocs, mesh, initsoln,
                                nregs=self.nregs, cfg=cfg)

        # Register index list and current index
        self._regidx = list(range(self.nregs))
        self._idxcurr = 0

        # Global degree of freedom count
        self._gndofs = self._get_gndofs()

        elementscls = self.system.elementscls
        self._subdims = [elementscls.convarmap[self.system.ndims].index(v)
                         for v in elementscls.dualcoeffs[self.system.ndims]]

        # Convergence tolerances
        self._pseudo_residtol = residtol = []
        for v in elementscls.convarmap[self.system.ndims]:
            try:
                residtol.append(cfg.getfloat(sect, 'pseudo-resid-tol-' + v))
            except NoOptionError:
                residtol.append(cfg.getfloat(sect, 'pseudo-resid-tol'))

        self._pseudo_norm = cfg.get(sect, 'pseudo-resid-norm', 'l2')
        if self._pseudo_norm not in {'l2', 'uniform'}:
            raise ValueError('Invalid pseudo-residual norm')

        # Plugins for the pseudo-integrator
        self.pseudo_plugins = self._get_pseudo_plugins()

        # Pointwise kernels for the pseudo-integrator
        self.pintgkernels = defaultdict(list)

        # Pseudo-step counter
        self.npseudosteps = 0

    @property
    def _pseudo_stepper_regidx(self):
        return self._regidx[:self.pseudo_stepper_nregs]

    @property
    def _source_regidx(self):
        sr = self.pseudo_stepper_nregs + self.stepper_nregs + self.stage_nregs
        return self._regidx[sr]

    @property
    def _stage_regidx(self):
        bsnregs = self.pseudo_stepper_nregs + self.stepper_nregs
        return self._regidx[bsnregs:bsnregs + self.stage_nregs]

    @property
    def _stepper_regidx(self):
        psnregs = self.pseudo_stepper_nregs
        return self._regidx[psnregs:psnregs + self.stepper_nregs]

    @property
    def _modes_regidx(self):
        cmnregs = self.pseudo_stepper_nregs + self.stepper_nregs + self.stage_nregs + 1 + self.modes_nregs
        return self._regidx[cmnregs:cmnregs + self.modes_nregs]

    @property
    def _pseudo_residual_regidx(self):
        r = self.pseudo_stepper_nregs + self.stepper_nregs + self.stage_nregs + 1 + self.modes_nregs
        return self._regidx[r]

    def init_stage(self, currstg, stepper_coeffs, dt):
        self.stepper_coeffs = stepper_coeffs
        self._dt = dt
        self.current_stage = currstg

        svals = [0, 1 / dt, *stepper_coeffs[:-1]]
        sregs = [self._source_regidx, *self._stepper_regidx,
                 *self._stage_regidx[:currstg]]

        # Accumulate physical stepper sources into a single register
        self._addv(svals, sregs, subdims=self._subdims)

    def finalise_stage(self, currstg, tcurr):
        if self.stage_nregs > 1:
            self.system.rhs(tcurr, self._idxcurr, self._stage_regidx[currstg])

    def store_current_soln(self):
        # Copy the current soln into the first source register
        self._add(0, self._stepper_regidx[0], 1, self._idxcurr)

    def obtain_solution(self, bcoeffs):
        consts = [0, 1, *bcoeffs]
        regidxs = [self._idxcurr, self._stepper_regidx[0], *self._stage_regidx]

        self._addv(consts, regidxs, subdims=self._subdims)

    def _get_pseudo_plugins(self):
        pseudo_plugins = []

        for s in self.cfg.sections():
            if (m := re.match('pseudo-plugin-(.+?)(?:-(.+))?$', s)):
                cfgsect, name, suffix = m[0], m[1], m[2]

                args = ('pseudo', name, self, cfgsect)
                args += (suffix, )

                data = {}

                # Instantiate
                pseudo_plugins.append(get_plugin(*args, **data))

        return pseudo_plugins

    def _run_pseudo_plugins(self):
        self.backend.wait()

        # Fire off the plugins and tally up the runtime
        for pseudo_plugin in self.pseudo_plugins:
            pseudo_plugin(self)

    def call_plugin_dtau(self, taus):
        self.taulist = deque(taus)

    def register_get(self,i):
        return self.system.ele_scal_upts(i)

    def extract_parameters(self, i):
        if not hasattr(self, 'parameters'):
            return {}
        
        parameters = {}
        
        for param_name, arr in self.parameters_s.items():
            if arr.shape[1] == 1:
                i = 0

            fin = arr[0,i]

            # If fin is not a float, then something is wrong
            if not isinstance(fin, float):
                raise ValueError('Invalid parameter value')

            parameters[param_name] = arr[0, i]

        return parameters

    def update_parameters(self, params):
        for param_name, arr in params.items():
            if param_name == 'zeta':
                self.system.ac_zeta = arr
