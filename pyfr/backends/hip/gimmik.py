from gimmik import HIPMatMul
import numpy as np

from pyfr.backends.base import NotSuitableError
from pyfr.backends.hip.provider import HIPKernel, HIPKernelProvider


class HIPGiMMiKKernels(HIPKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        # Maximum number of kernels to consider
        self.nkerns = backend.cfg.getint('backend-hip', 'gimmik-nkerns', 8)

        # Number of benchmarking runs
        self.nbench = backend.cfg.getint('backend-hip', 'gimmik-nbench', 5)

        # Kernel cache
        self._kerns = {}

        # Stream and events used for kernel benchmarking
        self._stream = backend.hip.create_stream()
        self._start_evt = backend.hip.create_event()
        self._stop_evt = backend.hip.create_event()

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        # Check that A is constant
        if 'const' not in a.tags:
            raise NotSuitableError('GiMMiK requires a constant a matrix')

        # Fetch the matrix and tally up the number of non-zeros
        arr = a.get()
        nnz, nuq = np.count_nonzero(arr), len(np.unique(np.abs(arr)))

        # Check that A is suitable
        if nuq > 28 and nnz / arr.size > 0.15:
            raise NotSuitableError('Matrix inappropriate GiMMiK')

        # Dimensions
        ldb, ldc = b.leaddim, out.leaddim

        # Alignment
        if 'align' in b.tags and 'align' in out.tags:
            aligne = self.backend.alignb // b.itemsize
        else:
            aligne = None

        # Cache key
        ckey = (a.mid, alpha, beta, aligne, ldb, ldc)

        # Check the kernel cache
        try:
            kern, grid, block = self._kerns[ckey]
        except KeyError:
            kname = f'gimmik_mm_{arr.shape[0]}x{arr.shape[1]}'
            stream, kdata = self._stream, None
            start_evt, stop_evt = self._start_evt, self._stop_evt
            best_dt, best_kern = None, None

            # Save a copy of the contents of the output matrix
            out_np = getattr(out, 'parent', out).get()

            mm = HIPMatMul(alpha*arr, beta=beta, aligne=aligne, n=b.ncol,
                           ldb=ldb, ldc=ldc)
            kgen = mm.kernels(a.dtype, kname=kname,
                              gcn_arch=self.backend.props['gcn_arch_name'])

            # Benchmark the sequence of kernels generated by GiMMiK
            try:
                for i in range(self.nkerns):
                    src, meta = kgen.send(kdata)
                    kern = self._build_kernel(kname, src, 'PP')

                    # Set the parameters
                    params = kern.make_params(meta['grid'], meta['block'])
                    params.set_args(b, out)

                    # Benchmark with warmup
                    for j in range(self.nbench + 1):
                        if j == 1:
                            start_evt.record(stream)

                        kern.exec_async(stream, params)

                    stop_evt.record(stream)
                    stream.synchronize()

                    dt = stop_evt.elapsed_time(start_evt)
                    if best_dt is None or dt < best_dt:
                        best_dt = dt
                        best_kern = kern, meta['grid'], meta['block']

                    kdata = {'runtime': dt}
            except StopIteration:
                pass

            # Restore the output matrix
            getattr(out, 'parent', out).set(out_np)

            # Update the cache
            self._kerns[ckey] = kern, grid, block = best_kern

        # Set the parameters
        params = kern.make_params(grid, block)
        params.set_args(b, out)

        class MulKernel(HIPKernel):
            def add_to_graph(self, graph, deps):
                pass

            def run(self, stream):
                kern.exec_async(stream, params)

        return MulKernel(mats=[b, out])
