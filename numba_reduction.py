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
        #reduction.allocate.end

        # reduction.kernel.begin
        @cuda.jit
        def array_sum(data, psum):
            tid = cuda.threadIdx.x
            size = len(data)
            if tid < size:
                i = cuda.grid(1)

                # Declare an array in shared memory
                shr = cuda.shared.array(shrsize,int32)
                shr[tid] = data[i]

                # Ensure writes to shared memory are visible
                # to all threads before reducing
                cuda.syncthreads()

                s = 1
                while s < cuda.blockDim.x:
                    if tid % (2 * s) == 0:
                        # Stride by `s` and add
                        shr[tid] += shr[tid + s]
                    s *= 2
                    cuda.syncthreads()

                # After the loop, the zeroth  element contains the sum
                if tid == 0:
                    psum[cuda.blockIdx.x] = shr[tid]
        # reduction.kernel.end

        # reduction.launch.begin
        nblocks = (nelem - 1) // nthreads + 1 
        b = cuda.to_device(np.arange(nblocks))
        array_sum[nblocks, nthreads](a,b)
        # reduction.launch.end

        # assert.begin
        for i in range(nblocks):
           np.testing.assert_equal(b[i], sum(np.arange(i*nthreads,(i+1)*nthreads)))
        # assert.end

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
