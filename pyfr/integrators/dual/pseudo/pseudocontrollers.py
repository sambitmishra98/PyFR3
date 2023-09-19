from time import perf_counter
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

    def _update_pseudostep_multipinfo(self, tcurr, *resids):
        self.pseudostep_multipinfo.append((self.ntotiters, tcurr, *resids))

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

        #self._init_isolate_mats()

        self.costs_sli = {}

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
    def register_isolate(self, l1, l1reg1, idx, idy, l1reg2):
        isolatek = []
        for i, a in enumerate(self.isolatemats[l1, idx, 0]):
            b = self.system.ele_banks[i][l1reg1]
            c = self.system.ele_banks[i][l1reg2]
            isolatek.append(self.backend.kernel('mul', a, b, out=c))

        return isolatek

    def isolateall(self, reg_to_isolate, regidxs_to_isolate_into):
        order = self.modes_nregs - 1

        for idx, mode_regid in enumerate(regidxs_to_isolate_into):
            self.backend.run_kernels(self.register_isolate(order, 
                                                           reg_to_isolate, 
                                                           idx, 0,  
                                                           mode_regid))


    def pseudo_advance(self, tcurr):
        self.tcurr = tcurr

        # Store the current register solution for later use
        solution_start = self.register(self._idxcurr)

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

        solution_end = self.register(self._idxcurr)
        difference = self.subtract(solution_start, solution_end)
        residual = self.divide(difference, self.dtau_mats)

        #if self.maxniters > 1:
        #    s_err = self.register(self._idxerr)

        # ----------------------------------------------------------------------
        # Use this space to store registers for the pseudo-plugin
        #if tcurr == 0 and self.ntotiters == 0:
        #    self.collected_registers[f'solution-start-{tcurr}']

        # ----------------------------------------------------------------------

        self.costs_sli['walltime'] = walltime

        for f in self.system.elementscls.convarmap[self.ndims]:
            self.costs_sli[f'res_l2-{f}'] = self.lin_op(self.extract(residual, f), 'l2')

        # Isolate modes of current and previous solutions idxcurr and idxprev
        #self.isolateall(self._idxprev, self._prev_modes_regidx)
        #self.isolateall(self._idxcurr, self._curr_modes_regidx)
        #resids = []
        #for ii in range(order+1):
        #    resids.append(self._resid(self._curr_modes_regidx[ii],
        #                              self._prev_modes_regidx[ii], 1))
        #self._update_pseudostep_multipinfo(tcurr, *resids)

    def extract_field(self, reg, field):
        if field not in self.system.elementscls.convarmap[self.ndims]:
            raise ValueError('Invalid field')

    def subtract(self, reg_1, reg_2):
        # Perform reg_1 - reg_2
        return [[np.subtract(inner_a, inner_b) for inner_a, inner_b in
                        zip(outer_a, outer_b)] for outer_a, outer_b in
                       zip(reg_1, reg_2)]

    def divide(self, reg_1, reg_2):
        # Perform reg_1 / reg_2
        return [[np.divide(inner_a, inner_b) for inner_a, inner_b in
                        zip(outer_a, outer_b)] for outer_a, outer_b in
                       zip(reg_1, reg_2)]

    def lin_op(self, reg, operation):
        if operation not in {'min', 'max', 'mean', 'std-dev', 'l1', 'l2', 'l-inf'}:
            raise ValueError('Invalid operation')

        if operation == 'min':
            return min([np.min(inner_array) for outer_array in 
                        reg for inner_array in outer_array])
        
        elif operation == 'max':
            return max([np.max(inner_array) for outer_array in 
                        reg for inner_array in outer_array])

        elif operation == 'mean':
            total_sum = sum([np.sum(inner_array) for outer_array in 
                             reg for inner_array in outer_array])
            total_elements = sum([inner_array.size for outer_array in 
                                  reg for inner_array in outer_array])
            return total_sum / total_elements
        
        elif operation == 'std-dev':
            mean_val = self.lin_op(reg, 'mean')
            total_elements = sum([inner_array.size for outer_array in 
                                  reg for inner_array in outer_array])
            squared_diff_sum = sum([np.sum((inner_array - mean_val) ** 2) 
                                    for outer_array in reg for inner_array in outer_array])
            return np.sqrt(squared_diff_sum / total_elements)

        elif operation == 'l1':
            return sum([np.sum(np.abs(inner_array)) for outer_array in reg 
                        for inner_array in outer_array])

        elif operation == 'l2':
            sum_of_squares = sum([np.sum(inner_array**2) for outer_array in reg 
                                  for inner_array in outer_array])
            return np.sqrt(sum_of_squares)
        
        elif operation == 'l-inf':
            return max([np.max(np.abs(inner_array)) for outer_array in reg 
                        for inner_array in outer_array])
            
    def extract(self, reg, field_variable):
        if field_variable not in {'p', 'u', 'v', 'w'}:
            raise ValueError('Invalid field variable')

        # Assuming 'p', 'u', and 'v' are in the order [p, u, v] in the innermost ndarray
        field_idx = {'p': 0, 'u': 1, 'v': 2, 'w': 3}[field_variable]

        # Extracting the specific field variable from each inner ndarray
        extracted = [[inner_array[field_idx] for inner_array in outer_array] for outer_array in reg]
        
        return extracted


    #def inspect_structure_2(self, data, depth=0, max_depth=2):
    #    if depth > max_depth:
    #        return '...'
    #    if isinstance(data, (list, tuple, np.ndarray)):
    #        return type(data).__name__ + '[' + ', '.join(self.inspect_structure_2(item, depth+1) for item in data) + ']'
    #    else:
    #        return str(type(data))
    #def inspect_data_structure_1(self, data, depth=0, max_depth=2):
    #    if depth > max_depth:
    #        return str(type(data))
    #    if isinstance(data, (list, tuple, np.ndarray)):
    #        element_types = {self.inspect_data_structure_1(item, depth + 1) for item in data}
    #        return f"{type(data)} of {element_types}"
    #    else:
    #        return str(type(data))
    #data_structure = self.inspect_data_structure_1(solution_start)  # replace 'your_variable' with the name of your variable
    #   print(data_structure)
