# NVSHMEM Device Atomic Memory Operations with Numba-CUDA DSL

This section documents the NVSHMEM Device Atomic Memory Operations with Numba-CUDA DSL.

## Example: Using atomic_add in a Numba-CUDA kernel

The following example demonstrates how to use the NVSHMEM atomic add operation (atomic_add) in a Numba-CUDA kernel. This allows multiple threads to safely increment a value in shared memory on a remote PE (processing element).

    import numpy as np
    import cupy as cp
    from numba import cuda
    import nvshmem
    import nvshmem.core.device.numba as nvshmem_numba
    from mpi4py import MPI

    @cuda.jit
    def atomic_add_kernel(dst, value, pe):
        tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        if tid == 0:
            # Atomically add 'value' to dst[0] on remote PE
            nvshmem_numba.atomic_add(dst, value, pe)

    # Initialize NVSHMEM
    dev = cudaDevice()
    dev.set_current()
    stream = dev.create_stream()
    nvshmem.init(dev=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi", stream=stream)

    # Get information about the current PE
    me = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    # Choose a remote PE (for example, next PE in a ring)
    pe = (me + 1) % n_pes

    # Allocate device buffer
    dst = nvshmem.array((1,), dtype=np.int32)
    # Launch kernel to perform atomic add on remote PE's dst[0]
    # Note, Numba-cuda does not accept a cuda.core Stream, so we need to pass the stream handle.
    atomic_add_kernel[1, 1](dst, 42, pe, stream=int(stream.handle))

    # Finalize NVSHMEM
    nvshmem.finalize(dev=dev, stream=stream)

This example atomically adds 42 to the dst[0] array on the next PE in a ring. Only thread 0 performs the atomic operation for demonstration purposes. In practice, you can have multiple threads performing atomic operations as needed.

For more details, see the Numba test suite and the NVSHMEM4Py documentation.

`nvshmem.core.device.numba.amo. atomic_inc ( dst , pe )`

Atomically increments `dst` by 1 (no return value).

Args:

  * `dst` (`Array`): Symmetric destination array on remote PE.
  * `pe` (`int`): Target PE.

`nvshmem.core.device.numba.amo. atomic_fetch_inc ( dst , pe )`

Atomically increments `dst` by 1 and returns the old value.

Args:

  * `dst` (`Array`): Symmetric destination array on remote PE.
  * `pe` (`int`): Target PE.

Returns:
    The value stored at `dst` prior to increment.

`nvshmem.core.device.numba.amo. atomic_fetch ( src , pe )`

Fetches the current value at symmetric `src` on PE `pe`.

Args:

  * `src` (`Array`): Symmetric source array on remote PE.
  * `pe` (`int`): PE to fetch from.

Returns:
    Current value stored at `src` on PE `pe`.

`nvshmem.core.device.numba.amo. atomic_set ( dst , value , pe )`

Sets the value at symmetric `dst` on PE `pe` to `value`.

Args:

  * `dst` (`Array`): Symmetric destination array on remote PE.
  * `value`: Value to set.
  * `pe` (`int`): Target PE.

`nvshmem.core.device.numba.amo. atomic_add ( dst , value , pe )`

Atomically adds `value` to `dst` (no return value).

Args:

  * `dst` (`Array`): Symmetric destination array on remote PE.
  * `value`: Value to add.
  * `pe` (`int`): Target PE.

`nvshmem.core.device.numba.amo. atomic_fetch_add ( dst , value , pe )`

Atomically adds `value` to `dst` and returns the old value.

Args:

  * `dst` (`Array`): Symmetric destination array on remote PE.
  * `value`: Value to add.
  * `pe` (`int`): Target PE.

Returns:
    The value stored at `dst` prior to the addition.

`nvshmem.core.device.numba.amo. atomic_and ( dst , value , pe )`

Atomically applies bitwise AND of `value` with `dst` (no return value).

Args:

  * `dst` (`Array`): Symmetric destination array on remote PE.
  * `value`: Value to AND with.
  * `pe` (`int`): Target PE.

`nvshmem.core.device.numba.amo. atomic_fetch_and ( dst , value , pe )`

Atomically applies bitwise AND and returns the old value.

Args:

  * `dst` (`Array`): Symmetric destination array on remote PE.
  * `value`: Value to AND with.
  * `pe` (`int`): Target PE.

Returns:
    The value stored at `dst` prior to the AND.

`nvshmem.core.device.numba.amo. atomic_or ( dst , value , pe )`

Atomically applies bitwise OR of `value` with `dst` (no return value).

Args:

  * `dst` (`Array`): Symmetric destination array on remote PE.
  * `value`: Value to OR with.
  * `pe` (`int`): Target PE.

`nvshmem.core.device.numba.amo. atomic_fetch_or ( dst , value , pe )`

Atomically applies bitwise OR and returns the old value.

Args:

  * `dst` (`Array`): Symmetric destination array on remote PE.
  * `value`: Value to OR with.
  * `pe` (`int`): Target PE.

Returns:
    The value stored at `dst` prior to the OR.

`nvshmem.core.device.numba.amo. atomic_xor ( dst , value , pe )`

Atomically applies bitwise XOR of `value` with `dst` (no return value).

Args:

  * `dst` (`Array`): Symmetric destination array on remote PE.
  * `value`: Value to XOR with.
  * `pe` (`int`): Target PE.

`nvshmem.core.device.numba.amo. atomic_fetch_xor ( dst , value , pe )`

Atomically applies bitwise XOR and returns the old value.

Args:

  * `dst` (`Array`): Symmetric destination array on remote PE.
  * `value`: Value to XOR with.
  * `pe` (`int`): Target PE.

Returns:
    The value stored at `dst` prior to the XOR.

`nvshmem.core.device.numba.amo. atomic_swap ( dst , value , pe )`

Atomically swaps the current value at `dst` with `value` on PE `pe`.

Args:

  * `dst` (`Array`): Symmetric destination array on remote PE.
  * `value`: Replacement value.
  * `pe` (`int`): Target PE.

Returns:
    The old value previously stored at `dst`.

`nvshmem.core.device.numba.amo. atomic_compare_swap ( dst , cond , value , pe )`

Atomically compares the current value at `dst` with `cond` and swaps with `value` if equal.

Args:

  * `dst` (`Array`): Symmetric destination location on remote PE.
  * `cond`: Comparison value.
  * `value`: Replacement value.
  * `pe` (`int`): Target PE.

Returns:
    The old value previously stored at `dst`.
