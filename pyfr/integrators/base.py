from collections import defaultdict, deque
import itertools as it
import re
import sys
from time import perf_counter

import numpy as np

from pyfr.cache import memoize
from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll
from pyfr.plugins import get_plugin

from pyfr.loadrelocator import LoadRelocator


def _common_plugin_prop(attr):
    def wrapfn(fn):
        @property
        def newfn(self):
            if not (p := getattr(self, attr)):
                t, c = perf_counter(), self._plugin_wtimes['common', None]
                p = fn(self)
                self._plugin_wtimes['common', None] = c + perf_counter() - t
                setattr(self, attr, p)

            return p
        return newfn
    return wrapfn


class BaseIntegrator:
    def __init__(self, backend, mesh, initsoln, cfg):
        self.backend = backend
        self.isrestart = initsoln is not None
        self.cfg = cfg
        self.prevcfgs = {f: initsoln[f].tostr() for f in initsoln or []
                         if f.startswith('config-')}

        # Start time
        self.tstart = cfg.getfloat('solver-time-integrator', 'tstart', 0.0)
        self.tend = cfg.getfloat('solver-time-integrator', 'tend')

        # Current time; defaults to tstart unless restarting
        if self.isrestart:
            stats = initsoln['stats']
            self.tcurr = stats.getfloat('solver-time-integrator', 'tcurr')
        else:
            self.tcurr = self.tstart

        # List of target times to advance to
        self.tlist = deque([self.tend])
        
        # List of steps with plugin pollution
        self.slist = []

        # Accepted and rejected step counters
        self.nacptsteps = 0
        self.nrjctsteps = 0
        self.nacptchain = 0

        # Current and minimum time steps
        self.dt = cfg.getfloat('solver-time-integrator', 'dt')
        self.dtmin = cfg.getfloat('solver-time-integrator', 'dt-min', 1e-12)

        # Extract the UUID of the mesh (to be saved with solutions)
        self.mesh_uuid = mesh.uuid

        if cfg.getbool('mesh', 'enable-relocator', False):
            comm, rank, root = get_comm_rank_root()
            self.observe_only = cfg.getbool('mesh', 'observe-only', True)

            if comm.size == 1 and not self.observe_only:
                raise ValueError('Relocation is only supported in parallel')

            self.tol = self.cfg.getfloat('mesh', 'imbalance-tol', 0.1)

            if not self.observe_only:
                self.load_relocator = LoadRelocator(mesh)

        self.lbdiff = 0.
        self.pprev  = 0.

        self._invalidate_caches()

        # Record the starting wall clock time
        self._wstart = perf_counter()

        # Record the total amount of time spent in each plugin
        self._plugin_wtimes = defaultdict(lambda: 0)

        # Abort computation
        self._abort = False
        self._abort_reason = ''

        # Smoothly step to target time in the last near_t steps
        self.aminf = self.cfg.getfloat('solver-time-integrator', 
                                          'dt-adjust-min-fact', 0.9)
        self.amaxf = self.cfg.getfloat('solver-time-integrator', 
                                          'dt-adjust-max-fact', 1.001)
        self.dt_fallback = cfg.getfloat('solver-time-integrator', 'dt')
        self.dt_near = None

    def adjust_dt(self, t):
        # Time difference to traverse 
        t_diff = t - self.tcurr

        # Estimate steps to reach t upon taking self.dt_fallback steps
        est_nsteps = t_diff / self.dt_fallback
        est_nsteps_roundup = -(est_nsteps // -self.amaxf)

        if est_nsteps_roundup == 1:
            # Exactly reach t
            self.dt = t_diff
            self.dt_near = None

        elif (est_nsteps - 1) / (est_nsteps_roundup - 1) < self.aminf:
            # Modify step to the approaching t

            dt_near = t_diff / est_nsteps_roundup

            if (self.dt_near is None 
                or not self.aminf < (self.dt_near/dt_near) < self.amaxf): 
                self.dt_near = dt_near

            self.dt = self.dt_near
        else:
            # Reset step if far from t
            self.dt = self.dt_fallback

    def plugin_abort(self, reason):
        self._abort = True
        self._abort_reason = self._abort_reason or reason

    def _get_plugins(self, initsoln):
        plugins = []

        for s in self.cfg.sections():
            if (m := re.match('(soln|solver)-plugin-(.+?)(?:-(.+))?$', s)):
                cfgsect, ptype, name, suffix = m[0], m[1], m[2], m[3]

                if ptype == 'solver' and suffix:
                    raise ValueError(f'solver-plugin-{name} cannot have a '
                                     'suffix')

                args = (ptype, name, self, cfgsect)
                if ptype == 'soln':
                    args += (suffix, )

                data = {}
                if initsoln is not None:
                    # Get the plugin data stored in the solution, if any
                    prefix = self.get_plugin_data_prefix(name, suffix)
                    for f in initsoln:
                        if f.startswith(f'{prefix}/'):
                            data[f.split('/')[2]] = initsoln[f]

                # Instantiate
                plugins.append(get_plugin(*args, **data))

        return plugins

    def _run_plugins(self):
        wtimes = self._plugin_wtimes

        self.backend.wait()

        # Fire off the plugins and tally up the runtime
        for plugin in self.plugins:
            tstart = perf_counter()
            tcommon = wtimes['common', None]

            plugin(self)

            dt = perf_counter() - tstart - wtimes['common', None] + tcommon

            pname = getattr(plugin, 'name', 'other')
            psuffix = getattr(plugin, 'suffix', None)
            wtimes[pname, psuffix] += dt

        # Abort if plugins request it
        self._check_abort()

    def _finalise_plugins(self):
        for plugin in self.plugins:
            if (finalise := getattr(plugin, 'finalise', None)):
                finalise(self)

    @staticmethod
    def get_plugin_data_prefix(name, suffix):
        if suffix:
            return f'plugins/{name}-{suffix}'
        else:
            return f'plugins/{name}'

    def call_plugin_dt(self, dt):
        ta = self.tlist
        tb = deque(np.arange(self.tend - dt, self.tcurr, -dt).tolist()[::-1])

        self.tlist = tlist = deque()

        # Merge the current and new time lists
        while ta and tb:
            t = ta.popleft() if ta[0] < tb[0] else tb.popleft()
            if not tlist or t - tlist[-1] > self.dtmin:
                tlist.append(t)

        for t in it.chain(ta, tb):
            if not tlist or t - tlist[-1] > self.dtmin:
                tlist.append(t)

    def call_plugin_nsteps(self, nsteps):
        self.slist.append(nsteps)

    def _invalidate_caches(self):
        self._curr_soln = None
        self._curr_grad_soln = None
        self._curr_dt_soln = None

    def step(self, t, dt):
        pass

    def advance_to(self, t):
        pass

    def run(self):
        for t in self.tlist:
            self.advance_to(t)

        self._finalise_plugins()

    @property
    def nsteps(self):
        return self.nacptsteps + self.nrjctsteps

    def collect_stats(self, stats):
        wtime = perf_counter() - self._wstart
        self._wtime = wtime

        # Simulation and wall clock times
        stats.set('solver-time-integrator', 'tcurr', self.tcurr)
        stats.set('solver-time-integrator', 'wall-time', wtime)

        # Plugin wall clock times
        for (pname, psuffix), t in self._plugin_wtimes.items():
            k = f'plugin-wall-time-{pname}'
            if psuffix:
                k += f'-{psuffix}'

            stats.set('solver-time-integrator', k, t)

        # Step counts
        stats.set('solver-time-integrator', 'nsteps', self.nsteps)
        stats.set('solver-time-integrator', 'nacptsteps', self.nacptsteps)
        stats.set('solver-time-integrator', 'nrjctsteps', self.nrjctsteps)

        # MPI wait times
        if self.cfg.getbool('backend', 'collect-wait-times', False):
            comm, rank, root = get_comm_rank_root()

            wait_times = comm.allgather(self.system.rhs_wait_times())
            for i, ms in enumerate(zip(*wait_times)):
                for j, k in enumerate(['mean', 'stdev', 'median']):
                    stats.set('backend-wait-times', f'rhs-graph-{i}-{k}',
                              ','.join(f'{v[j]:.3g}' for v in ms))

    @property
    def cfgmeta(self):
        cfg = self.cfg.tostr()

        if self.prevcfgs:
            ret = dict(self.prevcfgs, config=cfg)

            if cfg != ret[f'config-{len(self.prevcfgs) - 1}']:
                ret[f'config-{len(self.prevcfgs)}'] = cfg

            return ret
        else:
            return {'config': cfg, 'config-0': cfg}

    def _check_abort(self):
        comm, rank, root = get_comm_rank_root()

        if scal_coll(comm.Allreduce, int(self._abort), op=mpi.LOR):
            self._finalise_plugins()

            reason = self._abort_reason
            sys.exit(comm.allreduce(reason, op=lambda x, y: x or y))


class BaseCommon:
    def _get_gndofs(self):
        comm, rank, root = get_comm_rank_root()

        # Get the number of degrees of freedom in this partition
        ndofs = sum(self.system.ele_ndofs)

        # Sum to get the global number over all partitions
        return comm.allreduce(ndofs, op=mpi.SUM)

    @memoize
    def _get_axnpby_kerns(self, *rs, subdims=None):
        kerns = [self.backend.kernel('axnpby', *[em[r] for r in rs],
                                     subdims=subdims)
                 for em in self.system.ele_banks]

        return kerns

    @memoize
    def _get_reduction_kerns(self, *rs, **kwargs):
        dtau_mats = getattr(self, 'dtau_upts', [])

        kerns = []
        for em, dtaum in it.zip_longest(self.system.ele_banks, dtau_mats):
            kerns.append(self.backend.kernel('reduction', *[em[r] for r in rs],
                                             dt_mat=dtaum, **kwargs))

        return kerns

    def _addv(self, consts, regidxs, subdims=None):
        # Get a suitable set of axnpby kernels
        axnpby = self._get_axnpby_kerns(*regidxs, subdims=subdims)

        # Bind the arguments
        for k in axnpby:
            k.bind(*consts)

        self.backend.run_kernels(axnpby)

    def _add(self, *args, subdims=None):
        self._addv(args[::2], args[1::2], subdims=subdims)
