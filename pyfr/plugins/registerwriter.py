from pyfr.inifile import Inifile
from pyfr.mpiutil import get_comm_rank_root
from pyfr.plugins.base import BasePseudoPlugin, PostactionMixin, RegionMixin
from pyfr.writers.native import NativeWriter


class RegisterWriterPlugin(PostactionMixin, RegionMixin, BasePseudoPlugin):
    name = 'registerwriter'
    systems = ['*']
    formulations = ['dual']

    def __init__(self, intg, cfgsect, suffix=None):
        super().__init__(intg, cfgsect, suffix)

        # Base output directory and file name
        basedir = self.cfg.getpath(self.cfgsect, 'basedir', '.', abs=True)
        basename = self.cfg.get(self.cfgsect, 'basename')

        # Construct the solution writer
        self._writer = NativeWriter(intg, basedir, basename, 'soln')

        # Output field names
        self.fields = intg.system.elementscls.convarmap[self.ndims]

        # Output data type
        self.fpdtype = intg.backend.fpdtype

        # Register our output times with the integrator
        intg.collected_registers = {}

    def __call__(self, intg):

        comm, rank, root = get_comm_rank_root()

        # If the dictionary intg.collected_registers is empty then we have no work
        if not intg.collected_registers:
            return
        
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

        for file_name, register in intg.collected_registers.items():

        # Stack together expressions by element type
#        stacked_regs = [np.vstack(list(register)) 
#                           for register in zip(*registers)]

#        for (idx, etype, rgn), reg in zip(self._ele_regions, stacked_regs):
#            data[etype] = reg.astype(self.fpdtype)

            for idx, etype, rgn in self._ele_regions:
                data[etype] = register[idx][..., rgn].astype(self.fpdtype)

            # Write out the file
            solnfname = self._writer.write(data, intg.tcurr, metadata)

            intg.abort = False
            self._invoke_postaction(intg=intg, mesh=intg.system.mesh.fname,
                                    soln=solnfname, t=intg.tcurr)
