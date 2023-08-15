from collections import deque
import csv
from pprint import pprint
import numpy as np

from pyfr.integrators.dual.pseudo.base import BaseDualPseudoIntegrator
from pyfr.mpiutil import get_comm_rank_root, mpi


class BaseDualPseudoController(BaseDualPseudoIntegrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure the system is compatible with our formulation
        self.system.elementscls.validate_formulation(self)

        # Stats on the most recent step
        self.pseudostepinfo = []

        self.taulist = deque()

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

    def _show_resid(self, rcurr, rold, norm, dt_fac):
        comm, rank, root = get_comm_rank_root()

        # Get a set of kernels to compute the residual
        rkerns = self._get_reduction_kerns(rcurr, rold, method='resid',
                                           norm=norm)

        # Bind the dynmaic arguments
        for kern in rkerns:
            kern.bind(dt_fac)

        # Run the kernels
        self.backend.run_kernels(rkerns, wait=True)

        # Pseudo L2 norm
        if norm == 'l2':
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

    def wrapper_on_pseudo_plugins(self, i):
        if self.taulist:
            if (self.tcurr > self.taulist[0][0] and 
                self.current_stage == self.taulist[0][1] and 
                i == self.taulist[0][2]):

                self.taulist.pop()
                self._run_pseudo_plugins()

class DualNonePseudoController(BaseDualPseudoController):
    pseudo_controller_name = 'none'
    pseudo_controller_needs_lerrest = False

    def pseudo_advance(self, tcurr):
        self.tcurr = tcurr

        for i in range(self.maxniters):

            self.wrapper_on_pseudo_plugins(i)

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

        order = self.cfg.getint('solver', 'order')

        self.logfile = open(f'dtau-lvl{order}.csv', 'w', newline='')
        self.csvwriter = csv.writer(self.logfile)
        self.csvwriter.writerow(['tcurr', 'n', 
                                 'min-p', 'mean-p', 'max-p',
                                 'min-u', 'mean-u', 'max-u',
                                 ])

        self.logfile2 = open(f'res-lvl{order}.csv', 'w', newline='')
        self.csvwriter2 = csv.writer(self.logfile2)
        self.csvwriter2.writerow(['tcurr', 'n', 
                                 'l2-p', 'max-p',
                                 'l2-u', 'max-u',
                                 ])

        self.counter = 1

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
        tplargs['dtau_maxf'] = self.cfg.getfloat(sect, 'pseudo-dt-max-mult',
                                                 3.0)

        tplargs['dtau_minf'] = self.cfg.getfloat(sect, 'pseudo-dt-min-mult',
                                                 1.0)

        if not tplargs['minf'] < 1 <= tplargs['maxf']:
            raise ValueError('Invalid pseudo max-fact, min-fact')

        if tplargs['dtau_maxf'] < 1 < tplargs['dtau_minf']:
            raise ValueError('Invalid pseudo-dt-max-mult, pseudo-dt-max-mult')

        # Limits for the local pseudo-time-step size
        tplargs['dtau_min'] = self._dtau
        tplargs['dtau_max'] = tplargs['dtau_maxf'] * self._dtau
        tplargs['dtau_min'] = tplargs['dtau_minf'] * self._dtau

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

        self.backend.commit()

    def localerrest(self, errbank):
        self.backend.run_kernels(self.pintgkernels['localerrest', errbank])

    def init_stage(self, currstg, stepper_coeffs, dt):
        super().init_stage(currstg, stepper_coeffs, dt)
    
        # Update the pseudo time-step size
        #self.rewind_dtau()

    def pseudo_advance(self, tcurr):
        self.tcurr = tcurr

        for i in range(self.maxniters):
            # Take the step
            self._idxcurr, self._idxprev, self._idxerr = self.step(self.tcurr)
            self.localerrest(self._idxerr)

            #if i == 50:
            #    self.save_dtau()

            for dtau_mat in self.dtau_upts:
                mat = dtau_mat.get() # ~ (solution_points, nvars, elements)

                mins = mat.min(axis=(0,2))
                means = mat.mean(axis=(0,2))
                maxs = mat.max(axis=(0,2))
                
                self.csvwriter.writerow([self.tcurr  , self.counter, 
                                            mins[0], means[0], maxs[0],
                                            mins[1], means[1], maxs[1],
                                         ])

            norm_l2    = self._show_resid(self._idxcurr, self._idxprev, 'l2', 1)
            norm_l_inf = self._show_resid(self._idxcurr, self._idxprev, 'max', 1)
            self.csvwriter2.writerow([self.tcurr  , self.counter, 
                                    norm_l2[0], norm_l_inf[0], 
                                    norm_l2[1], norm_l_inf[1], 
                                    ])

            self.counter += 1

            self.wrapper_on_pseudo_plugins(i)
            if self.convmon(i, self.minniters):
                break
