from collections import defaultdict

import numpy as np

from pyfr.integrators.dual.pseudo.base import BaseDualPseudoIntegrator
from pyfr.util import memoize
from pyfr.mpiutil import get_comm_rank_root, mpi


class BaseDualPseudoController(BaseDualPseudoIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure the system is compatible with our formulation
        self.system.elementscls.validate_formulation(self)

        # Stats on the most recent step
        self.pseudostepinfo = []

        # Stats on every multip level for the most recent step
        self.pseudostep_multipinfo = []

    def convmon(self, i, minniters, dt_fac=1):
        if i >= minniters - 1:
            # Compute the normalised residual
            resid = self._resid(self._idxcurr, self._idxprev, dt_fac)

            self._update_pseudostepinfo(i + 1, resid)
            return all(r <= t for r, t in zip(resid, self._pseudo_residtol))
        else:
            self._update_pseudostepinfo(i + 1, None)
            return False

    def commit(self):
        self.system.commit()

    def _resid(self, rcurr, rold, dt_fac):
        comm, rank, root = get_comm_rank_root()

        # Get a set of kernels to compute the residual
        rkerns = self._get_reduction_kerns(rcurr, rold, method='resid',
                                           norm=self._pseudo_norm)

        # Bind the dynmaic arguments
        for kern in rkerns:
            kern.bind(dt_fac)

        # Run the kernels
        self.backend.run_kernels(rkerns, wait=True)

        # Pseudo L2 norm
        if self._pseudo_norm == 'l2':
            # Reduce locally (element types) and globally (MPI ranks)
            res = np.array([sum(e) for e in zip(*[r.retval for r in rkerns])])
            comm.Allreduce(mpi.IN_PLACE, res, op=mpi.SUM)

            # Normalise and return
            return tuple(np.sqrt(res / self._gndofs))
        # Uniform norm
        else:
            # Reduce locally (element types) and globally (MPI ranks)
            res = np.array([max(e) for e in zip(*[r.retval for r in rkerns])])
            comm.Allreduce(mpi.IN_PLACE, res, op=mpi.MAX)

            # Normalise and return
            return tuple(np.sqrt(res))

    def _update_pseudostepinfo(self, niters, resid):
        self.pseudostepinfo.append((self.ntotiters, niters, resid))

    def _update_pseudostep_multipinfo(self, order, niters, resids):
        self.pseudostep_multipinfo.append((self.ntotiters, order, niters, resids))


class DualNonePseudoController(BaseDualPseudoController):
    pseudo_controller_name = 'none'
    pseudo_controller_needs_lerrest = False

    def pseudo_advance(self, tcurr):
        self.tcurr = tcurr

        for i in range(self.maxniters):
            # Take the step
            self._idxcurr, self._idxprev = self.step(self.tcurr)

            # Convergence monitoring
            if self.convmon(i, self.minniters, self._dtau):
                break


class DualPIPseudoController(BaseDualPseudoController):
    pseudo_controller_name = 'local-pi'
    pseudo_controller_needs_lerrest = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sect = 'solver-time-integrator'

        # Error norm
        self._norm = self.cfg.get(sect, 'errest-norm', 'l2')
        if self._norm not in {'l2', 'uniform'}:
            raise ValueError('Invalid error norm')

        tplargs = {'nvars': self.system.nvars}

        # Error tolerance
        tplargs['atol'] = self.cfg.getfloat(sect, 'atol')

        # PI control values
        sord = self.pseudo_stepper_order
        tplargs['expa'] = self.cfg.getfloat(sect, 'pi-alpha', 0.7) / sord
        tplargs['expb'] = self.cfg.getfloat(sect, 'pi-beta', 0.4) / sord

        # Constants
        tplargs['maxf'] = self.cfg.getfloat(sect, 'max-fact', 1.01)
        tplargs['minf'] = self.cfg.getfloat(sect, 'min-fact', 0.98)
        tplargs['saff'] = self.cfg.getfloat(sect, 'safety-fact', 0.8)

        tplargs['dtau_minf'] = self.cfg.getfloat(sect, 'pseudo-dt-min-mult',
                                                 0.1)
        tplargs['dtau_maxf'] = self.cfg.getfloat(sect, 'pseudo-dt-max-mult',
                                                 3.0)

        tplargs['dtau_minf_p'] = self.cfg.getfloat(sect, 'pseudo-dt-min-mult-p',
                                                 tplargs['dtau_minf'])
        tplargs['dtau_maxf_p'] = self.cfg.getfloat(sect, 'pseudo-dt-max-mult-p',
                                                 tplargs['dtau_maxf'])

        if not tplargs['minf'] < 1 <= tplargs['maxf']:
            raise ValueError('Invalid pseudo max-fact, min-fact')

        # Limits for the local pseudo-time-step size
        tplargs['dtau_min'] = tplargs['dtau_minf'] * self._dtau
        tplargs['dtau_max'] = tplargs['dtau_maxf'] * self._dtau

        tplargs['dtau_min_p'] = tplargs['dtau_minf'] * self._dtau 
        tplargs['dtau_max_p'] = tplargs['dtau_maxf'] * self._dtau 

        # Register a kernel to compute local error
        self.backend.pointwise.register(
            'pyfr.integrators.dual.pseudo.kernels.localerrest'
        )

        for ele, shape, dtaumat in zip(self.system.ele_map.values(),
                                       self.system.ele_shapes, self.dtau_upts):
            # Allocate storage for previous error
            err_prev = self.backend.matrix(shape, np.ones(shape),
                                           tags={'align'})

            # Append the error kernels to the list
            for i, err in enumerate(ele.scal_upts):
                self.pintgkernels['localerrest', i].append(
                    self.backend.kernel(
                        'localerrest', tplargs=tplargs,
                        dims=[ele.nupts, ele.neles], err=err,
                        errprev=err_prev, dtau_upts=dtaumat
                    )
                )

        self._init_isolate_mats()

        print(
                f" pseudo_stepper = {self._pseudo_stepper_regidx}\t"
                f" stepper        = {self._stepper_regidx       }\t"
                f" stage          = {self._stage_regidx         }\t"
                f" source         = {self._source_regidx        }\t"
                f" prev_modes     = {self._prev_modes_regidx    }\t"
                f" prev_modes     = {self._curr_modes_regidx    }\t"
                )


        self.backend.commit()

    def localerrest(self, errbank):
        self.backend.run_kernels(self.pintgkernels['localerrest', errbank])

    def _init_isolate_mats(self):
        self.isolatemats = defaultdict(list)
        cmat = lambda m: self.backend.const_matrix(m, tags={'align'})

        order = self.modes_nregs - 1

        for etype in self.system.ele_types:
            b = self.system.ele_map[etype].basis.ubasis
            for level_to_isolate in range(order+1):
                self.isolatemats[order, level_to_isolate].append(cmat(b.isolate(level_to_isolate)))

    @memoize
    def register_isolate(self, l1, l1reg1, level_to_isolate, l1reg2):
        isolatek = []
        for i, a in enumerate(self.isolatemats[l1, level_to_isolate]):
            b = self.system.ele_banks[i][l1reg1]
            c = self.system.ele_banks[i][l1reg2]
            isolatek.append(self.backend.kernel('mul', a, b, out=c))

        return isolatek

    def isolateall(self, what_to_isolate, where_to_isolate):
        order = self.modes_nregs - 1
        for level_to_isolate, mode_regid in enumerate(where_to_isolate):
            self.backend.run_kernels(self.register_isolate(order, what_to_isolate, level_to_isolate, mode_regid))

    def pseudo_advance(self, tcurr):
        self.tcurr = tcurr
        order = self.modes_nregs - 1

        for i in range(self.maxniters):
            # Take the step
            self._idxcurr, self._idxprev, self._idxerr = self.step(self.tcurr)
            self.localerrest(self._idxerr)

            if self.convmon(i, self.minniters):
                break

        # Isolate modes of current and previous solutions idxcurr and idxprev
        self.isolateall(self._idxprev, self._prev_modes_regidx)
        self.isolateall(self._idxcurr, self._curr_modes_regidx)

        resids = []
        for ii in range(order+1):
            resids.append(self._resid(self._curr_modes_regidx[ii], self._prev_modes_regidx[ii], 1))

        self._update_pseudostep_multipinfo(order, i + 1, resids)

#        print(f"{tcurr = }, {i = }, {order = } \t", end='')
#        p_res_rms = 0
#        for i in range(order+1):
#            p_res = self._resid(self._curr_modes_regidx[i], self._prev_modes_regidx[i], 1)[1]
#            print(f"L{i} = {p_res} \t", end='')
#            p_res_rms += p_res**2
#        print(f"ALL: {self._resid(self._idxcurr, self._idxprev, 1)[1]} \t", end='')
#        print(f"DIFF: {self._resid(self._idxcurr, self._idxprev, 1)[1] - np.sqrt(p_res_rms)} \t")
