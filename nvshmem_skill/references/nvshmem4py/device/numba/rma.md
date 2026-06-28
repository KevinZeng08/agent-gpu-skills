# NVSHMEM Device Remote Memory Access (RMA) with Numba-CUDA DSL

This section documents the NVSHMEM Device Remote Memory Access (RMA) operations with Numba-CUDA DSL.

## Example: Using put and get in a Numba-CUDA kernel

The following example demonstrates how to use the NVSHMEM put and get operations in a Numba-CUDA kernel. These allow threads to write to and read from memory on a remote PE (processing element) directly from device code.

    import numpy as np
    import cupy as cp
    from numba import cuda
    import nvshmem
    import nvshmem.core.device.numba as nvshmem_numba
    from mpi4py import MPI

    @cuda.jit
    def rma_kernel(src, dst, remote_buf, pe):
        tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        if tid == 0:
            # Put data from src to remote_buf on remote PE
            nvshmem_numba.put(remote_buf, src, pe)
            # Get data from remote_buf on remote PE to dst
            nvshmem_numba.get(dst, remote_buf, pe)

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

    # Allocate device buffers
    src = nvshmem.array((1,), dtype=np.int32)
    dst = nvshmem.array((1,), dtype=np.int32)
    remote_buf = nvshmem.array((1,), dtype=np.int32)

    # Launch kernel to perform put and get on remote PE's buffer
    # Note, Numba-cuda does not accept a cuda.core Stream, so we need to pass the stream handle.
    rma_kernel[1, 1](src, dst, remote_buf, pe, stream=int(stream.handle))

    # Finalize NVSHMEM
    nvshmem.finalize(dev=dev, stream=stream)

This example puts the value from src[0] to the remote_buf[0] on the next PE in a ring, and then gets the value back into dst[0]. Only thread 0 performs the RMA operations for demonstration purposes. In practice, you can have multiple threads performing RMA as needed.

For more details, see the Numba test suite and the NVSHMEM4Py documentation.

`nvshmem.core.device.numba.rma. p ( )`

Put immediate data from src to dst on PE pe. Device initiated.

`src` must be a scalar value, passed as a symmetric Array of size 1.

Args:

  * `dst` (`Array`): Destination symmetric array on remote PE.
  * `src` (`Array`): Scalar source value (size-1 array) on this PE.
  * `pe` (`int`): PE to put to.

`nvshmem.core.device.numba.rma. g ( )`

Get immediate data from src on PE pe to local dst. Device initiated.

`src` must be a scalar value, passed as a symmetric Array of size 1.

Args:

  * `src` (`Array`): Source symmetric array (size-1) on remote PE.
  * `pe` (`int`): PE to get from.

Returns:
    Scalar value retrieved from `src` on PE `pe`.

`nvshmem.core.device.numba.rma. put ( src: numba.core.types.npytypes.Array , dst: numba.core.types.npytypes.Array , pe: int32 ) → None`

Copies data from local `src` to symmetric `dst` on PE `pe`. This is a thread-level operation.

Args:

  * `src` (`Array`): Local source array on this PE to copy from.
  * `dst` (`Array`): Symmetric destination array on PE `pe` to copy to.
  * `pe` (`int`): PE to copy to.

Note:
    When `src` and `dst` are of different sizes, only data of the smaller size is copied. `src` and `dst` must have the same data type.

`nvshmem.core.device.numba.rma. get ( dst: numba.core.types.npytypes.Array , src: numba.core.types.npytypes.Array , pe: int32 ) → None`

Copies data from symmetric `src` on PE `pe` to local `dst`. This is a thread-level operation.

Args:

  * `dst` (`Array`): Local destination array on this PE to copy to.
  * `src` (`Array`): Symmetric source array from PE `pe` to copy from.
  * `pe` (`int`): PE to copy from.

Note:
    When `src` and `dst` are of different sizes, only data of the smaller size is copied. `src` and `dst` must have the same data type.

`nvshmem.core.device.numba.rma. put_nbi ( src: numba.core.types.npytypes.Array , dst: numba.core.types.npytypes.Array , pe: int32 ) → None`

Non-blockingly copies data from local `src` to symmetric `dst` on PE `pe`. This is a thread-level operation.

Args:

  * `src` (`Array`): Local source array on this PE to copy from.
  * `dst` (`Array`): Symmetric destination array on PE `pe` to copy to.
  * `pe` (`int`): PE to copy to.

Note:
    When `src` and `dst` are of different sizes, only data of the smaller size is copied. `src` and `dst` must have the same data type.

`nvshmem.core.device.numba.rma. get_nbi ( dst: numba.core.types.npytypes.Array , src: numba.core.types.npytypes.Array , pe: int32 ) → None`

Non-blockingly copies data from symmetric `src` on PE `pe` to local `dst`. This is a thread-level operation.

Args:

  * `dst` (`Array`): Local destination array on this PE to copy to.
  * `src` (`Array`): Symmetric source array from PE `pe` to copy from.
  * `pe` (`int`): PE to copy from.

Note:
    When `src` and `dst` are of different sizes, only data of the smaller size is copied. `src` and `dst` must have the same data type.

`nvshmem.core.device.numba.rma. put_block ( src: numba.core.types.npytypes.Array , dst: numba.core.types.npytypes.Array , pe: int32 ) → None`

Copies from local `src` to symmetric `dst` on PE `pe`. This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

Args:

  * `src` (`Array`): Local source array on this PE to copy from.
  * `dst` (`Array`): Symmetric destination array on PE `pe` to copy to.
  * `pe` (`int`): PE to copy to.

Note:
    When `src` and `dst` are of different sizes, only data of the smaller size is copied. `src` and `dst` must have the same data type.

`nvshmem.core.device.numba.rma. get_block ( dst: numba.core.types.npytypes.Array , src: numba.core.types.npytypes.Array , pe: int32 ) → None`

Copies from symmetric `src` on PE `pe` to local `dst`. This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

Args:

  * `dst` (`Array`): Local destination array on this PE to copy to.
  * `src` (`Array`): Symmetric source array from PE `pe` to copy from.
  * `pe` (`int`): PE to copy from.

Note:
    When `src` and `dst` are of different sizes, only data of the smaller size is copied. `src` and `dst` must have the same data type.

`nvshmem.core.device.numba.rma. put_nbi_block ( src: numba.core.types.npytypes.Array , dst: numba.core.types.npytypes.Array , pe: int32 ) → None`

Non-blockingly copies from local `src` to symmetric `dst` on PE `pe`. This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

Args:

  * `src` (`Array`): Local source array on this PE to copy from.
  * `dst` (`Array`): Symmetric destination array on PE `pe` to copy to.
  * `pe` (`int`): PE to copy to.

`nvshmem.core.device.numba.rma. get_nbi_block ( dst: numba.core.types.npytypes.Array , src: numba.core.types.npytypes.Array , pe: int32 ) → None`

Non-blockingly copies from symmetric `src` on PE `pe` to local `dst`. This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

Args:

  * `dst` (`Array`): Local destination array on this PE to copy to.
  * `src` (`Array`): Symmetric source array from PE `pe` to copy from.
  * `pe` (`int`): PE to copy from.

`nvshmem.core.device.numba.rma. put_warp ( src: numba.core.types.npytypes.Array , dst: numba.core.types.npytypes.Array , pe: int32 ) → None`

Copies from local `src` to symmetric `dst` on PE `pe`. This is a warp-level operation. All threads in the warp must call this function with the same arguments.

Args:

  * `src` (`Array`): Local source array on this PE to copy from.
  * `dst` (`Array`): Symmetric destination array on PE `pe` to copy to.
  * `pe` (`int`): PE to copy to.

Note:
    When `src` and `dst` are of different sizes, only data of the smaller size is copied. `src` and `dst` must have the same data type.

`nvshmem.core.device.numba.rma. get_warp ( dst: numba.core.types.npytypes.Array , src: numba.core.types.npytypes.Array , pe: int32 ) → None`

Copies from symmetric `src` on PE `pe` to local `dst`. This is a warp-level operation. All threads in the warp must call this function with the same arguments.

Args:

  * `dst` (`Array`): Local destination array on this PE to copy to.
  * `src` (`Array`): Symmetric source array from PE `pe` to copy from.
  * `pe` (`int`): PE to copy from.

Note:
    When `src` and `dst` are of different sizes, only data of the smaller size is copied. `src` and `dst` must have the same data type.

`nvshmem.core.device.numba.rma. put_nbi_warp ( src: numba.core.types.npytypes.Array , dst: numba.core.types.npytypes.Array , pe: int32 ) → None`

Non-blockingly copies from local `src` to symmetric `dst` on PE `pe`. This is a warp-level operation. All threads in the warp must call this function with the same arguments.

Args:

  * `src` (`Array`): Local source array on this PE to copy from.
  * `dst` (`Array`): Symmetric destination array on PE `pe` to copy to.
  * `pe` (`int`): PE to copy to.

Note:
    When `src` and `dst` are of different sizes, only data of the smaller size is copied. `src` and `dst` must have the same data type.

`nvshmem.core.device.numba.rma. get_nbi_warp ( dst: numba.core.types.npytypes.Array , src: numba.core.types.npytypes.Array , pe: int32 ) → None`

Non-blockingly copies from symmetric `src` on PE `pe` to local `dst`. This is a warp-level operation. All threads in the warp must call this function with the same arguments.

Args:

  * `dst` (`Array`): Local destination array on this PE to copy to.
  * `src` (`Array`): Symmetric source array from PE `pe` to copy from.
  * `pe` (`int`): PE to copy from.

Note:
    When `src` and `dst` are of different sizes, only data of the smaller size is copied. `src` and `dst` must have the same data type.

`nvshmem.core.device.numba.rma. put_signal_block ( )`

Put data with a signal operation from `src` to `dst` on PE `pe` in a manner at a scope. Device initiated.

Signal variables must be an `Array` of dtype `int64` (8 bytes) allocated by or registered with NVSHMEM4Py. Supported signal operations are `SignalOp.SIGNAL_SET` and `SignalOp.SIGNAL_ADD`.

Args:

  * `dst` (`Array`): Destination symmetric array on remote PE.
  * `src` (`Array`): Source symmetric array on this PE.
  * `signal_var` (`Array`): Symmetric signal variable (dtype `int64`).
  * `signal_val` (`int`): Signal value.
  * `signal_op` (`SignalOp`): Signal operation type.
  * `pe` (`int`): Target PE to put to.

`nvshmem.core.device.numba.rma. put_signal ( )`

Put data with a signal operation from `src` to `dst` on PE `pe` in a manner at a scope. Device initiated.

Signal variables must be an `Array` of dtype `int64` (8 bytes) allocated by or registered with NVSHMEM4Py. Supported signal operations are `SignalOp.SIGNAL_SET` and `SignalOp.SIGNAL_ADD`.

Args:

  * `dst` (`Array`): Destination symmetric array on remote PE.
  * `src` (`Array`): Source symmetric array on this PE.
  * `signal_var` (`Array`): Symmetric signal variable (dtype `int64`).
  * `signal_val` (`int`): Signal value.
  * `signal_op` (`SignalOp`): Signal operation type.
  * `pe` (`int`): Target PE to put to.

`nvshmem.core.device.numba.rma. put_signal_nbi ( )`

Put data with a signal operation from `src` to `dst` on PE `pe` in a _nbi manner at a scope. Device initiated.

Signal variables must be an `Array` of dtype `int64` (8 bytes) allocated by or registered with NVSHMEM4Py. Supported signal operations are `SignalOp.SIGNAL_SET` and `SignalOp.SIGNAL_ADD`.

Args:

  * `dst` (`Array`): Destination symmetric array on remote PE.
  * `src` (`Array`): Source symmetric array on this PE.
  * `signal_var` (`Array`): Symmetric signal variable (dtype `int64`).
  * `signal_val` (`int`): Signal value.
  * `signal_op` (`SignalOp`): Signal operation type.
  * `pe` (`int`): Target PE to put to.

`nvshmem.core.device.numba.rma. put_signal_warp ( )`

Put data with a signal operation from `src` to `dst` on PE `pe` in a manner at a scope. Device initiated.

Signal variables must be an `Array` of dtype `int64` (8 bytes) allocated by or registered with NVSHMEM4Py. Supported signal operations are `SignalOp.SIGNAL_SET` and `SignalOp.SIGNAL_ADD`.

Args:

  * `dst` (`Array`): Destination symmetric array on remote PE.
  * `src` (`Array`): Source symmetric array on this PE.
  * `signal_var` (`Array`): Symmetric signal variable (dtype `int64`).
  * `signal_val` (`int`): Signal value.
  * `signal_op` (`SignalOp`): Signal operation type.
  * `pe` (`int`): Target PE to put to.

`nvshmem.core.device.numba.rma. put_signal_nbi_block ( )`

Put data with a signal operation from `src` to `dst` on PE `pe` in a _nbi manner at a scope. Device initiated.

Signal variables must be an `Array` of dtype `int64` (8 bytes) allocated by or registered with NVSHMEM4Py. Supported signal operations are `SignalOp.SIGNAL_SET` and `SignalOp.SIGNAL_ADD`.

Args:

  * `dst` (`Array`): Destination symmetric array on remote PE.
  * `src` (`Array`): Source symmetric array on this PE.
  * `signal_var` (`Array`): Symmetric signal variable (dtype `int64`).
  * `signal_val` (`int`): Signal value.
  * `signal_op` (`SignalOp`): Signal operation type.
  * `pe` (`int`): Target PE to put to.

`nvshmem.core.device.numba.rma. put_signal_nbi_warp ( )`

Put data with a signal operation from `src` to `dst` on PE `pe` in a _nbi manner at a scope. Device initiated.

Signal variables must be an `Array` of dtype `int64` (8 bytes) allocated by or registered with NVSHMEM4Py. Supported signal operations are `SignalOp.SIGNAL_SET` and `SignalOp.SIGNAL_ADD`.

Args:

  * `dst` (`Array`): Destination symmetric array on remote PE.
  * `src` (`Array`): Source symmetric array on this PE.
  * `signal_var` (`Array`): Symmetric signal variable (dtype `int64`).
  * `signal_val` (`int`): Signal value.
  * `signal_op` (`SignalOp`): Signal operation type.
  * `pe` (`int`): Target PE to put to.

`nvshmem.core.device.numba.rma. put_signal_nbi_warp ( )`

Put data with a signal operation from `src` to `dst` on PE `pe` in a _nbi manner at a scope. Device initiated.

Signal variables must be an `Array` of dtype `int64` (8 bytes) allocated by or registered with NVSHMEM4Py. Supported signal operations are `SignalOp.SIGNAL_SET` and `SignalOp.SIGNAL_ADD`.

Args:

  * `dst` (`Array`): Destination symmetric array on remote PE.
  * `src` (`Array`): Source symmetric array on this PE.
  * `signal_var` (`Array`): Symmetric signal variable (dtype `int64`).
  * `signal_val` (`int`): Signal value.
  * `signal_op` (`SignalOp`): Signal operation type.
  * `pe` (`int`): Target PE to put to.

# NVSHMEM4Py Memory Management with Numba-CUDA DSL

This section documents the NVSHMEM4Py Memory Management with Numba-CUDA DSL.

NVSHMEM4Py provides functions to access remote symmetric and multicast buffers as CuPy arrays via the nvshmem.core.device.numba.mem module.

`nvshmem.core.device.numba.mem. get_multicast_array ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , array: numba.core.types.npytypes.Array )`

Returns an array view on multicast-accessible memory corresponding to the input array. The Array passed into it must be allocated by NVSHMEM4Py.

This is the Python array equivalent of nvshmemx_mc_ptr which returns a pointer into a peer’s symmetric heap

Args:
    array: Array - A symmetric array allocated by NVSHMEM team: Teams - A NVSHMEM Team to create the Multicast object across
Returns:

  * A Numba ArrayView which represents the Multicast object

NOTE: This function is only supported with the Numba Runtime. To enable it,
    set the environment variable `NUMBA_CUDA_ENABLE_NRT=1`

`nvshmem.core.device.numba.mem. get_peer_array ( arr: numba.core.types.npytypes.Array , pe: int )`

Returns an array view of a peer buffer associated with an NVSHMEM-allocated object.

This is the Python array equivalent of nvshmem_ptr which returns a pointer into a peer’s symmetric heap

Args:
    array: Array - A symmetric array allocated by NVSHMEM pe: int - The remote PE to retrieve an ArrayView into
Returns:

  * A Numba ArrayView which represents the peer object

NOTE: This function is only supported with the Numba Runtime. To enable it,
    set the environment variable `NUMBA_CUDA_ENABLE_NRT=1`
