from time import perf_counter, sleep
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


class DualNonePseudoController(BaseDualPseudoController):
    pseudo_controller_name = 'none'
    pseudo_controller_needs_lerrest = False

    def pseudo_advance(self, tcurr):
        self.tcurr = tcurr

        for i in range(self.maxniters):

            params = self.extract_parameters(i)
            self.update_parameters(params)

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
        tplargs['atol'] = self.cfg.getfloat(sect, 'atol', 1e-10)

        # PI control values
        sord = self.pseudo_stepper_order
        tplargs['expa'] = self.cfg.getfloat(sect, 'pi-alpha', 0.7) / sord
        tplargs['expb'] = self.cfg.getfloat(sect, 'pi-beta', 0.4) / sord

        # Constants
        tplargs['maxf'] = self.cfg.getfloat(sect, 'max-fact', 1.01)
        tplargs['minf'] = self.cfg.getfloat(sect, 'min-fact', 0.98)
        tplargs['saff'] = self.cfg.getfloat(sect, 'safety-fact', 0.8)
        tplargs['dtau_maxf'] = self.cfg.getfloat(sect, 'pseudo-dt-max-mult',
                                                 10000.0)
        tplargs['dtau_minf'] = self.cfg.getfloat(sect, 'pseudo-dt-min-mult',
                                                 0.0001)

        if not tplargs['minf'] < 1 <= tplargs['maxf']:
            raise ValueError('Invalid pseudo max-fact, min-fact')

        if tplargs['dtau_maxf'] < 1 <= tplargs['dtau_minf']:
            raise ValueError('Invalid pseudo-dt-max-mult')

        # Limits for the local pseudo-time-step size
        tplargs['dtau_min'] = self._dtau
        tplargs['dtau_max'] = tplargs['dtau_maxf'] * self._dtau

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

        self.costs_sli = {}

        self.backend.commit()

    def localerrest(self, errbank):
        self.backend.run_kernels(self.pintgkernels['localerrest', errbank])

    def _init_isolate_mats(self):
        self.isolatemats = defaultdict(list)
        cmat = lambda m: self.backend.const_matrix(m, tags={'align'})

        # We have order = self.modes_nregs squared
        order = int((self.modes_nregs - 1)/2)

        for etype in self.system.ele_types:
            b = self.system.ele_map[etype].basis.ubasis
            for idx in range(self.modes_nregs):
                if etype == 'quad':
                    self.isolatemats[order, idx].append(
                                    cmat(b.isolate(idx)))
                else:
                    self.isolatemats[order, idx].append(
                                    cmat(b.zeros()))

    @memoize
    def register_isolate(self, l1, l1reg1, idx, l1reg2):
        isolatek = []

        for i, a in enumerate(self.isolatemats[l1, idx]):
            b = self.system.ele_banks[i][l1reg1]
            c = self.system.ele_banks[i][l1reg2]

            isolatek.append(self.backend.kernel('mul', a, b, out=c))

        return isolatek

    def isolateall(self, reg_to_isolate, regidxs_to_isolate_into):
        order = int((self.modes_nregs - 1)/2)

        for i, mode_regid in enumerate(regidxs_to_isolate_into):
            self.backend.run_kernels(self.register_isolate(order, reg_to_isolate, 
                                                           i, mode_regid))

    def pseudo_advance(self, tcurr):
        self.tcurr = tcurr

        # Store the current register solution for later use
        solution_start = self.system.ele_scal_upts(self._idxcurr)

        walltime = 0.
        for i in range(self.maxniters):

            params = self.extract_parameters(i)
            self.update_parameters(params)

            # Take the step

            walltime_start = perf_counter()
            self._idxcurr, self._idxprev, self._idxerr = self.step(self.tcurr)
            walltime += (perf_counter() - walltime_start)

            self.localerrest(self._idxerr)

            if self.convmon(i, self.minniters):
                break

        solution_end = self.system.ele_scal_upts(self._idxcurr)
        difference = self.subtract(solution_end, solution_start)
        residual = self.divide(difference, self.dtau_mats)
        self.system.ele_scal_upts_set(self._pseudo_residual_regidx, residual)
        # ---------------------------------------------------------------------
        # CORRECT UNTILL HERE
        # ---------------------------------------------------------------------

        self.isolateall(self._pseudo_residual_regidx, self._modes_regidx)

        self.costs_sli['walltime'] = walltime
        for f in self.system.elementscls.convarmap[self.ndims]:
            self.costs_sli[f'res_l2-{f}'] = self.lin_op(self.extract(residual, f), 'l2')

        for f in self.system.elementscls.convarmap[self.ndims]:
            vector_of_residual_modes = np.zeros(3)
            for i, mode_id in enumerate(self._modes_regidx):
                extracted_mode = self.extract(self.system.ele_scal_upts(mode_id), f)
                vector_of_residual_modes[i] = self.lin_op(extracted_mode, 'l2')

            self.costs_sli[f'res_modes_l2-{f}'] = vector_of_residual_modes
            
    def subtract(self, reg_1, reg_2):
        return [np.subtract(elem1, elem2) for elem1, elem2 in zip(reg_1, reg_2)]

    def divide(self, reg_1, reg_2):
        return [np.divide(elem1, elem2) for elem1, elem2 in zip(reg_1, reg_2)]

    def lin_op(self, reg, operation):
        if operation not in {'min', 'max', 'mean', 'std-dev', 'l1', 'l2', 'l-inf'}:
            raise ValueError('Invalid operation')

        if operation == 'min':
            return min([np.min(elem) for elem in reg])
            
        elif operation == 'max':
            return max([np.max(elem) for elem in reg])

        elif operation == 'mean':
            total_sum = sum([np.sum(elem) for elem in reg])
            total_elements = sum([elem.size for elem in reg])
            return total_sum / total_elements
            
        elif operation == 'std-dev':
            mean_val = self.lin_op(reg, 'mean')
            total_elements = sum([elem.size for elem in  reg])
            squared_diff_sum = sum([np.sum((elem - mean_val) ** 2) for elem in  reg])
            return np.sqrt(squared_diff_sum / total_elements)

        elif operation == 'l1':
            return sum([np.sum(np.abs(elem)) for elem in reg])

        elif operation == 'l2':
            sum_of_squares = sum([np.sum(np.square(elem)) for elem in reg])
            return np.sqrt(sum_of_squares)
            
        elif operation == 'l-inf':
            return max([np.max(np.abs(elem)) for elem in reg])
            
    def extract(self, reg, field_variable):
        if field_variable not in {'p', 'u', 'v'}:
            raise ValueError('Invalid field variable')

        # Mapping for 'p', 'u', and 'v' based on their order
        field_idx = {'p': 0, 'u': 1, 'v': 2, 'w': 3}[field_variable]

        # Extracting the specific field variable from each inner ndarray but preserving the middle dimension
        extracted = [elem[:,field_idx,:] for elem in reg]

        return extracted
