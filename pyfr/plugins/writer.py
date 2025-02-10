import numpy as np

from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.base import (BaseSolnPlugin, LoadBalanceMixin,
                               PostactionMixin, RegionMixin)
from pyfr.writers.native import NativeWriter
from pyfr.util import first


class WriterPlugin(PostactionMixin, RegionMixin, LoadBalanceMixin,
                   BaseSolnPlugin):
    name = 'writer'
    systems = ['*']
    formulations = ['dual', 'std']
    dimensions = [2, 3]

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Base output directory and file name
        basedir = self.cfg.getpath(cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(cfgsect, 'basename')

        # Get the element map and region data
        self._emap, erdata = intg.system.ele_map, self._ele_region_data

        # Decide if gradients should be written or not
        self._write_grads = self.cfg.getbool(cfgsect, 'write-gradients', False)

        # Figure out the shape of each element type in our region
        self._nvars = self.nvars + self._write_grads*(self.nvars*self.ndims)
        ershapes = {etype: (self._nvars, self._emap[etype].nupts) for etype in erdata}

        # Construct the solution writer
        self._writer = NativeWriter.from_integrator(intg, basedir, basename,
                                                    'soln')
        self._writer.set_shapes_eidxs(ershapes, erdata)

        # Asynchronous output options
        self._async_timeout = self.cfg.getfloat(cfgsect, 'async-timeout', 60)

        # Output time step and last output time
        self.dt_out = self.cfg.getfloat(cfgsect, 'dt-out')
        self.tout_last = intg.tcurr

        # Output field names
        self.fields = list(first(intg.system.ele_map.values()).convars)
        if self._write_grads:
            dims = 'xyz'[:self.ndims]
            self.fields += [f'grad_{f}_{d}' for f in self.fields for d in dims]

        # Output data type
        self.fpdtype = intg.backend.fpdtype

        # Register our output times with the integrator
        intg.call_plugin_dt(self.dt_out)
        intg.called_plugin_dt = True

        # If we're not restarting then make sure we write out the initial
        # solution when we are called for the first time
        if not intg.isrestart:
            self.tout_last -= self.dt_out

    def _prepare_metadata(self, intg):
        comm, rank, root = get_comm_rank_root()

        stats = Inifile()
        stats.set('data', 'fields', ','.join(self.fields))
        stats.set('data', 'prefix', 'soln')
        intg.collect_stats(stats)

        # If we are the root rank then prepare the metadata
        if rank == root:
            metadata = {**intg.cfgmeta, 'stats': stats.tostr(),
                        'mesh-uuid': intg.mesh_uuid}
        else:
            metadata = None

        # Fetch data from other plugins and add it to metadata with ad-hoc keys
        for csh in intg.plugins:
            try:
                prefix = intg.get_plugin_data_prefix(csh.name, csh.suffix)
                pdata = csh.serialise(intg)
            except AttributeError:
                pdata = {}

            if rank == root:
                metadata |= {f'{prefix}/{k}': v for k, v in pdata.items()}

        return metadata

    def _prepare_data(self, intg):
        data = {}

        if self._write_grads:
            if hasattr(intg, 'load_relocator'):
                soln = self.recreate_pmesh_ary(intg, intg.soln,
                                                    self.src_mmesh,
                                                    self.dest_mmesh)
                grad_soln = self.recreate_pmesh_ary(intg, intg.grad_soln,
                                                    self.src_mmesh,
                                                    self.dest_mmesh)
            else:
                soln, grad_soln = intg.soln, intg.grad_soln
        else:
            if hasattr(intg, 'load_relocator'):
                soln = self.recreate_pmesh_ary(intg, intg.soln,
                                                    self.src_mmesh,
                                                    self.dest_mmesh)
                grad_soln = None
            else:
                soln, grad_soln = intg.soln, None

        for idx, etype, rgn in self._ele_regions:
            d = soln[idx][..., rgn].T.astype(self.fpdtype)

            if self._write_grads:
                g = grad_soln[idx][..., rgn].transpose(3, 2, 0, 1)
                g = g.reshape(len(g), self.ndims*self.nvars, -1)

                d = np.hstack([d, g], dtype=self.fpdtype)

            data[etype] = d

        return data

    def __call__(self, intg):
        comm, rank, root = get_comm_rank_root()

        self._writer.probe()

        if intg.tcurr - self.tout_last < self.dt_out - self.tol:
            return

        if comm != mpi.COMM_NULL:
            if hasattr(intg, 'load_relocator'):
                if self.partition == 'balanced':
                    intg.load_relocator.mm.connect_mmeshes('plugins_new', 
                                                        'compute_new',
                                                        overwrite=True)
                elif self.partition == 'base':
                    intg.load_relocator.mm.connect_mmeshes('base', 
                                                        'compute_new',
                                                        overwrite=True)
                elif self.partition == 'compute':
                    # Re-tally ecounts for new compute-mesh
                    self._writer._ecounts = {etype: len(
                            self.mesh.eidxs.get(etype, [])
                            ) for etype in self.mesh.etypes}

                    self.redo_ele_region_data(intg)
                    ershapes = {etype: (self._nvars, self._emap[etype].nupts) 
                                        for etype in self._ele_region_data}
                    self._writer.set_shapes_eidxs(ershapes, self._ele_region_data)

            # Prepare the data and metadata
            data = self._prepare_data(intg)
            metadata = self._prepare_metadata(intg)

            # Prepare a callback to kick off any postactions
            callback = lambda fname, t=intg.tcurr: self._invoke_postaction(
                intg=intg, mesh=self.mesh.fname, soln=fname, t=t
            )

            # Write out the file
            self._writer.write(data, intg.tcurr, metadata, self._async_timeout,
                            callback)

        # Update the last output time
        self.tout_last = intg.tcurr

    def finalise(self, intg):
        super().finalise(intg)

        self._writer.flush()
