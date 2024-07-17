from collections import defaultdict, deque
import itertools as it
import os
import re
import subprocess
import sys
import time

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, mpi, scal_coll
from pyfr.plugins import get_plugin
from pyfr.util import memoize


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

        # Accepted and rejected step counters
        self.nacptsteps = 0
        self.nrjctsteps = 0
        self.nacptchain = 0

        # Current and minimum time steps
        self._dt = cfg.getfloat('solver-time-integrator', 'dt')
        self.dtmin = cfg.getfloat('solver-time-integrator', 'dt-min', 1e-12)

        # Extract the UUID of the mesh (to be saved with solutions)
        self.mesh_uuid = mesh.uuid

        self._invalidate_caches()

        # Record the starting wall clock time
        self._wstart = time.time()
        self._wprev = self.wtime = 0.

        # Record the total amount of time spent in each plugin
        self._plugin_wtimes = defaultdict(lambda: 0)

        # Abort computation
        self._abort = False
        self._abort_reason = ''

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
        self.backend.wait()

        # Fire off the plugins and tally up the runtime
        for plugin in self.plugins:
            tstart = time.time()

            plugin(self)

            pname = getattr(plugin, 'name', 'other')
            psuffix = getattr(plugin, 'suffix', None)
            self._plugin_wtimes[pname, psuffix] += time.time() - tstart

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
        tb = deque(np.arange(self.tcurr, self.tend, dt).tolist())

        self.tlist = tlist = deque()

        # Merge the current and new time lists
        while ta and tb:
            t = ta.popleft() if ta[0] < tb[0] else tb.popleft()
            if not tlist or t - tlist[-1] > self.dtmin:
                tlist.append(t)

        for t in it.chain(ta, tb):
            if not tlist or t - tlist[-1] > self.dtmin:
                tlist.append(t)

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

    def collect_a_device_on_node(self, stats):

        # Get device name at rank 0. This would suffice for homogenous nodes.
        if self.backend.name == 'cuda':
            device_name = subprocess.check_output(
                ["lspci", "-d", "10de::0300", "-nn"], universal_newlines=True)
        elif self.backend.name == 'hip':
            device_name = subprocess.check_output(
                ["lspci", "-d", "1002::0300", "-nn"], universal_newlines=True)
        elif self.backend.name == 'opencl':
            device_name = self.backend.device_name
        elif self.backend.name == 'openmp':
             # Get the device name using lscpu
            device_name = subprocess.check_output(
                ["lscpu"], universal_newlines=True).split('\n')
            device_name = [x for x in device_name if 'Model name' in x]
            device_name = device_name[0].split(':')[-1].strip()
        else:
            raise ValueError(f"Unknown backend {self.backend.name}")

        stats.set('simulation', f'rank-0', device_name)

        return stats

    def collect_sim_details(self, sim_rep_stats):

#        sim_rep_stats = self.collect_device_on_node(sim_rep_stats)

        comm, rank, root = get_comm_rank_root()

        sim_rep_stats = self.collect_a_device_on_node(sim_rep_stats)

        # Get the mesh file name
        sim_rep_stats.set('simulation', 'mesh-file', self.system.mesh.fname)

        ndofs_total = 0
        for etype, *ndofs in self.system.etype_ndofs.items():
            ndof = sum(ndofs)
            ndofs = comm.allreduce(*ndofs, op=mpi.SUM)
            sim_rep_stats.set('simulation', f'ndofs-{etype}', ndofs)
            ndofs_total += ndofs
        sim_rep_stats.set('simulation', 'ndofs-total', ndofs_total)

        # Name the set of simulations
        sim_rep_stats.set('configuration', 'test-name', None)

        # Save the time of execution
        sim_rep_stats.set('simulation', 'time-of-execution',
                            time.strftime('%Y-%m-%d %H:%M:%S'))

        # Print directory that the code is present in
        cwd = os.path.dirname(os.path.abspath(__file__))
        cwd = os.path.abspath(os.path.join(cwd,
                                           os.pardir, os.pardir))

        # Expose the exact version of source code.
        source_code_label = subprocess.check_output(
            ["git", "describe", "--always", "--abbrev=40"], cwd=cwd)
        sim_rep_stats.set('simulation', 'source-code-git-label',
                  source_code_label.strip().decode('utf-8'))

        # Save the environment variables that are set
        for k, v in os.environ.items():
            try:
                # Get all the environment variables that match
                sim_rep_stats.set('environment', k, v)

                if 'SLURM_JOB_ID' in k:

                    # Get the job id
                    job_id = v

                    # Store the job script using a subprocess: scontrol write batch_script ${SLURM_JOB_ID} -
                    job_script = subprocess.check_output(
                        ["scontrol", "write", "batch_script", job_id, "-"], universal_newlines=True)
                    sim_rep_stats.set('slurm-job', 'script', job_script)

            except ValueError:
                pass

        return sim_rep_stats

    def collect_kernel_benchmark_details(self, benchmark_stats):
        # Add to the benchmark_stats
        for k, v in self.backend.bench_kern.items():

            benchmark_stats.set('kernels', k, v)

    #        2*row['nnz']*row['b.ncol']/row['Runtime']/1e12

            if 'Runtime' in k:

                kname = k.split('_')[:-1]
                kname = '_'.join(kname)

                nnz = self.backend.bench_kern[kname + '_nnz']
                b = self.backend.bench_kern[kname + '_b-cols']

                performance = 2*nnz*b/v

                benchmark_stats.set('benchmark', kname, performance)

        return benchmark_stats

    def collect_stats(self, stats):
        wtime = time.time() - self._wstart

        # Simulation and wall clock times
        stats.set('solver-time-integrator', 'tcurr', self.tcurr)
        stats.set('solver-time-integrator', 'wall-time', wtime)

        # Plugin wall clock times
        for (pname, psuffix), t in self._plugin_wtimes.items():
            k = f'plugin-wall-time-{pname}'
            if psuffix:
                k += f'-{psuffix}'

            stats.set('solver-time-integrator', k, t)

        # Pythonic way to add up all the plugin wall times
        plugin_wtime_total = sum(self._plugin_wtimes.values())

        # Extract wall-time spent towards compute only
        self.wdiff = wtime - plugin_wtime_total - self._wprev
        self._wprev = wtime - plugin_wtime_total

        stats = self.collect_sim_details(stats)

        stats = self.collect_kernel_benchmark_details(stats)

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
