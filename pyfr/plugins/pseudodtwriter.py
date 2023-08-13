from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BasePseudoPlugin, PostactionMixin, RegionMixin
from pyfr.writers.native import NativeWriter


class PseudoDtWriterPlugin(PostactionMixin, RegionMixin, BasePseudoPlugin):
    name = 'pseudodtwriter'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Base output directory and file name
        basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(self.cfgsect, 'basename')

        # Construct the solution writer
        self._writer = NativeWriter(intg, basedir, basename, 'soln')

        # Get list of times at which to output
        # Give details of time, stage, pseudo-iteration
        # Format: [(1, 1, 1),]
        self.times = self.cfg.getliteral(self.cfgsect, 'times')

        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]

        # Output data type
        self.fpdtype = intg.backend.fpdtype

        # Register our output times with the integrator
        intg.call_plugin_dtau(self.times)

    def __call__(self, intg):

        comm, rank, root = get_comm_rank_root()

        stats = Inifile()
        stats.set('data', 'fields', ','.join(self.fields))
        stats.set('data', 'prefix', 'soln')
        stats.set('solver-time-integrator', 'tcurr', str(intg.tcurr))
        intg.collect_stats(stats)

        # If we are the root rank then prepare the metadata
        if rank == root:
            metadata = dict(intg.cfgmeta,
                            stats=stats.tostr(),
                            mesh_uuid=intg.mesh_uuid)
        else:
            metadata = None

        data = dict(self._ele_region_data)

        for idx, etype, rgn in self._ele_regions:
            vals = intg.dtau_mats[idx]
            data[etype] = vals[..., rgn].astype(self.fpdtype)

        # Write out the file
        solnfname = self._writer.write(data, intg.tcurr, metadata)

        intg.abort = False
        self._invoke_postaction(intg=intg, mesh=intg.system.mesh.fname,
                                soln=solnfname, t=intg.tcurr)

