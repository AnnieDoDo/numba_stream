import unittest

from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout


@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestReduction(CUDATestCase):
    """
    Test shared memory reduction
    """

    def setUp(self):
        # Prevent output from this test showing up when running the test suite
        self._captured_stdout = captured_stdout()
        self._captured_stdout.__enter__()
        super().setUp()

    def tearDown(self):
        # No exception type, value, or traceback
        self._captured_stdout.__exit__(None, None, None)
        super().tearDown()

    def ex_reduction(self, insize):
        import numpy as np
        from numba import cuda
        from numba.types import int32

        # reduction.allocate.begin
        # generate data
        ndata = insize
        nthreads = 256
        shrsize = nthreads*2
        a = cuda.to_device(np.arange(ndata))
        nelem = len(a)

        nstreams = 2
        chunk_size = (insize + nstreams - 1) // nstreams
        streams = [cuda.stream() for _ in range(nstreams)]
        partial_sums = cuda.device_array(nstreams, dtype=np.int32)
        #reduction.allocate.end

        # reduction.kernel.begin
        @cuda.jit
        def array_sum(data, psum):

        # Parameters
            nthreads = 256
            shrsize = nthreads * 2

            # Input data
            a = np.arange(insize, dtype=np.int32)
            d_a = cuda.to_device(a)

            # Partition input data across streams
            chunk_size = (insize + nstreams - 1) // nstreams
            streams = [cuda.stream() for _ in range(nstreams)]
            partial_sums = cuda.device_array(nstreams, dtype=np.int32)

            # Reduction kernel
            @cuda.jit
            def array_sum(data, psum, offset, size):
                tid = cuda.threadIdx.x
                bid = cuda.blockIdx.x
                i = offset + bid * cuda.blockDim.x + tid

                # Declare shared memory
                shr = cuda.shared.array(256, int32)

                # Load data into shared memory
                if i < offset + size:
                    shr[tid] = data[i]
                else:
                    shr[tid] = 0  # Handle out-of-bounds threads

                # Synchronize threads
                cuda.syncthreads()

                # Perform reduction in shared memory
                stride = 1
                while stride < cuda.blockDim.x:
                    if tid % (2 * stride) == 0:
                        shr[tid] += shr[tid + stride]
                    stride *= 2
                    cuda.syncthreads()

                # Write block sum to global memory
                if tid == 0:
                    psum[bid] = shr[0]

            # Launch reduction kernels in multiple streams
            for i, stream in enumerate(streams):
                offset = i * chunk_size
                size = min(chunk_size, insize - offset)
                nblocks = (size + nthreads - 1) // nthreads

                if size > 0:
                    array_sum[nblocks, nthreads, stream](d_a, partial_sums, offset, size)

            # Synchronize all streams
            for stream in streams:
                stream.synchronize()

            # Copy partial sums to host and validate
            h_partial_sums = partial_sums.copy_to_host()
            final_sum = np.sum(h_partial_sums)

            # Validate the result
            expected_sum = np.sum(a)
            np.testing.assert_equal(final_sum, expected_sum)

    def test_reduction_256(self):
        self.ex_reduction(256)

    def test_reduction_1024(self):
        self.ex_reduction(1024)

    def test_reduction_10240(self):
        self.ex_reduction(10240)

    def test_reduction_20480(self):
        self.ex_reduction(20480)

    #def test_reduction_10241(self):
    #    self.ex_reduction(10241)

if __name__ == "__main__":
    unittest.main()
