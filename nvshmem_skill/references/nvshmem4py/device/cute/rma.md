# NVSHMEM Device Remote Memory Access (RMA) with CuTe DSL

This section documents the NVSHMEM Device Remote Memory Access (RMA) operations with CuTe DSL.

## Example: Using `put` and `get` in a CuTe kernel

The following example demonstrates how to use the NVSHMEM `put` and `get` operations in a CuTe kernel. These allow threads to write to and read from memory on a remote PE (processing element) directly from device code.

    import cutlass
    from cutlass import cute
    from cuda.core import Device, Stream
    import nvshmem
    import nvshmem.core.device.cute as nvshmem_cute
    import nvshmem.core.interop.cute as nvshmem_cute_interop
    from mpi4py import MPI

    @cute.kernel
    def rma_kernel(src: cute.Tensor, dst: cute.Tensor, remote_buf: cute.Tensor, pe: cutlass.Int32):
        # Put data from src to remote_buf on remote PE
        nvshmem_cute.put_block(remote_buf, src, pe)
        # Get data from remote_buf on remote PE to dst
        nvshmem_cute.get_block(dst, remote_buf, pe)

    @cute.jit
    def rma_launcher(src, dst, remote_buf, pe):
        rma_kernel[1, 1](src, dst, remote_buf, pe)

    # Initialize NVSHMEM
    dev = Device()
    dev.set_current()
    stream = dev.create_stream()
    nvshmem.init(dev=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi", stream=stream)

    # Get information about the current PE
    me = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    # Choose a remote PE (for example, next PE in a ring)
    pe = (me + 1) % n_pes

    # Allocate device tensors using NVSHMEM symmetric memory
    src = nvshmem_cute_interop.tensor((1,), dtype=cute.Int32)
    dst = nvshmem_cute_interop.tensor((1,), dtype=cute.Int32)
    remote_buf = nvshmem_cute_interop.tensor((1,), dtype=cute.Int32)

    # Compile and launch the kernel
    compiled_fn, nvshmem_kernel = nvshmem_cute_interop.cute_compile_helper(
        rma_launcher, src, dst, remote_buf, pe
    )
    compiled_fn(src, dst, remote_buf, pe, stream=stream)

    # Finalize NVSHMEM
    nvshmem.core.library_finalize(nvshmem_kernel)
    nvshmem_cute_interop.cleanup_cute()
    nvshmem.finalize(dev=dev, stream=stream)

This example puts the value from `src` to the `remote_buf` on the next PE in a ring, and then gets the value back into `dst`. The CTA-level `put_block` and `get_block` operations require all threads in the CTA to call with the same arguments.

`nvshmem.core.device.cute.rma. p ( dst , src , pe )`

Writes a single scalar value `src` to the symmetric location `dst` on PE `pe`. This is a thread-level point operation (scalar put).

Unlike `put`, which transfers an array of elements, `p` transfers exactly one scalar value. `dst` must point to a single-element symmetric location.

Args:

  * `dst`: CuTe tensor view pointing to a single-element symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor with exactly one element.
  * `src`: Scalar value to write. The value is cast to the dtype of `dst`.
  * `pe` (`int`): Target PE.

Note:
    This is a blocking, thread-level operation. `src` is cast to the element dtype of `dst` before the transfer.

`nvshmem.core.device.cute.rma. g ( src , pe )`

Reads and returns a single scalar value from the symmetric location `src` on PE `pe`. This is a thread-level get operation (scalar get).

Unlike `get`, which transfers an array of elements, `g` retrieves exactly one scalar value. `src` must point to a single-element symmetric location.

Args:

  * `src`: CuTe tensor view pointing to a single-element symmetric source on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor with exactly one element.
  * `pe` (`int`): Source PE to read from.

Returns:
    The scalar value stored at `src` on PE `pe`, with the same dtype as `src`.
Note:
    This is a blocking, thread-level operation. The returned value is immediately available.

`nvshmem.core.device.cute.rma. put ( dst , src , pe )`

Copies data from local `src` to symmetric `dst` on PE `pe`. This is a thread-level operation.

Args:

  * `dst`: CuTe tensor view pointing to the symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
  * `src`: CuTe tensor view pointing to the local source data on this PE.
  * `pe` (`int`): Target PE to copy to.

Note:
    The number of elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a blocking operation: local data movement completes before the call returns.

`nvshmem.core.device.cute.rma. get ( dst , src , pe )`

Copies data from symmetric `src` on PE `pe` to local `dst`. This is a thread-level operation.

Args:

  * `dst`: CuTe tensor view pointing to the local destination on this PE.
  * `src`: CuTe tensor view pointing to the symmetric source on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
  * `pe` (`int`): Source PE to copy from.

Note:
    The number of elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a blocking operation: data is available in `dst` before the call returns.

`nvshmem.core.device.cute.rma. put_nbi ( dst , src , pe )`

Non-blockingly copies data from local `src` to symmetric `dst` on PE `pe`. This is a thread-level operation.

Args:

  * `dst`: CuTe tensor view pointing to the symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
  * `src`: CuTe tensor view pointing to the local source data on this PE.
  * `pe` (`int`): Target PE to copy to.

Note:
    The number of elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a non-blocking operation: the transfer may not be complete when the call returns. Use a fence or synchronization primitive to ensure completion.

`nvshmem.core.device.cute.rma. get_nbi ( dst , src , pe )`

Non-blockingly copies data from symmetric `src` on PE `pe` to local `dst`. This is a thread-level operation.

Args:

  * `dst`: CuTe tensor view pointing to the local destination on this PE.
  * `src`: CuTe tensor view pointing to the symmetric source on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
  * `pe` (`int`): Source PE to copy from.

Note:
    The number of elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a non-blocking operation: the transfer may not be complete when the call returns. Use a fence or synchronization primitive to ensure completion.

`nvshmem.core.device.cute.rma. put_block ( dst , src , pe )`

Copies data from local `src` to symmetric `dst` on PE `pe`. This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

Args:

  * `dst`: CuTe tensor view pointing to the symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
  * `src`: CuTe tensor view pointing to the local source data on this PE.
  * `pe` (`int`): Target PE to copy to.

Note:
    The number of elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a blocking operation: local data movement completes before the call returns.

`nvshmem.core.device.cute.rma. get_block ( dst , src , pe )`

Copies data from symmetric `src` on PE `pe` to local `dst`. This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

Args:

  * `dst`: CuTe tensor view pointing to the local destination on this PE.
  * `src`: CuTe tensor view pointing to the symmetric source on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
  * `pe` (`int`): Source PE to copy from.

Note:
    The number of elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a blocking operation: data is available in `dst` before the call returns.

`nvshmem.core.device.cute.rma. put_nbi_block ( dst , src , pe )`

Non-blockingly copies data from local `src` to symmetric `dst` on PE `pe`. This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

Args:

  * `dst`: CuTe tensor view pointing to the symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
  * `src`: CuTe tensor view pointing to the local source data on this PE.
  * `pe` (`int`): Target PE to copy to.

Note:
    The number of elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a non-blocking operation: the transfer may not be complete when the call returns. Use a fence or synchronization primitive to ensure completion.

`nvshmem.core.device.cute.rma. get_nbi_block ( dst , src , pe )`

Non-blockingly copies data from symmetric `src` on PE `pe` to local `dst`. This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

Args:

  * `dst`: CuTe tensor view pointing to the local destination on this PE.
  * `src`: CuTe tensor view pointing to the symmetric source on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
  * `pe` (`int`): Source PE to copy from.

Note:
    The number of elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a non-blocking operation: the transfer may not be complete when the call returns. Use a fence or synchronization primitive to ensure completion.

`nvshmem.core.device.cute.rma. put_warp ( dst , src , pe )`

Copies data from local `src` to symmetric `dst` on PE `pe`. This is a warp-level operation. All threads in the warp must call this function with the same arguments.

Args:

  * `dst`: CuTe tensor view pointing to the symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
  * `src`: CuTe tensor view pointing to the local source data on this PE.
  * `pe` (`int`): Target PE to copy to.

Note:
    The number of elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a blocking operation: local data movement completes before the call returns.

`nvshmem.core.device.cute.rma. get_warp ( dst , src , pe )`

Copies data from symmetric `src` on PE `pe` to local `dst`. This is a warp-level operation. All threads in the warp must call this function with the same arguments.

Args:

  * `dst`: CuTe tensor view pointing to the local destination on this PE.
  * `src`: CuTe tensor view pointing to the symmetric source on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
  * `pe` (`int`): Source PE to copy from.

Note:
    The number of elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a blocking operation: data is available in `dst` before the call returns.

`nvshmem.core.device.cute.rma. put_nbi_warp ( dst , src , pe )`

Non-blockingly copies data from local `src` to symmetric `dst` on PE `pe`. This is a warp-level operation. All threads in the warp must call this function with the same arguments.

Args:

  * `dst`: CuTe tensor view pointing to the symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
  * `src`: CuTe tensor view pointing to the local source data on this PE.
  * `pe` (`int`): Target PE to copy to.

Note:
    The number of elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a non-blocking operation: the transfer may not be complete when the call returns. Use a fence or synchronization primitive to ensure completion.

`nvshmem.core.device.cute.rma. get_nbi_warp ( dst , src , pe )`

Non-blockingly copies data from symmetric `src` on PE `pe` to local `dst`. This is a warp-level operation. All threads in the warp must call this function with the same arguments.

Args:

  * `dst`: CuTe tensor view pointing to the local destination on this PE.
  * `src`: CuTe tensor view pointing to the symmetric source on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor accessible by all PEs.
  * `pe` (`int`): Source PE to copy from.

Note:
    The number of elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a non-blocking operation: the transfer may not be complete when the call returns. Use a fence or synchronization primitive to ensure completion.

`nvshmem.core.device.cute.rma. put_signal_block ( dst , src , signal_var , signal_val , signal_op , pe )`

Puts data from `src` to symmetric `dst` on PE `pe`, then signals `signal_var`. This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

The signal operation atomically updates `signal_var` on the remote PE after the data transfer, allowing the remote PE to detect transfer completion via a signal variable.

Args:

  * `dst`: CuTe tensor view pointing to the symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor.
  * `src`: CuTe tensor view pointing to the local source data on this PE.
  * `signal_var`: CuTe tensor view pointing to a symmetric signal variable (dtype `uint64`) on PE `pe`. Must be a 1-element NVSHMEM-allocated tensor.
  * `signal_val` (`int`): Value used to update the signal variable. Cast to `uint64`.
  * `signal_op`: Signal operation type. Supported values are `NVSHMEM_SIGNAL_SET` (set the signal to `signal_val`) and `NVSHMEM_SIGNAL_ADD` (atomically add `signal_val` to the signal variable).
  * `pe` (`int`): Target PE.

Note:
    The number of data elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a blocking operation: the data transfer and signal update complete before the call returns.

`nvshmem.core.device.cute.rma. put_signal ( dst , src , signal_var , signal_val , signal_op , pe )`

Puts data from `src` to symmetric `dst` on PE `pe`, then signals `signal_var`. This is a thread-level operation.

The signal operation atomically updates `signal_var` on the remote PE after the data transfer, allowing the remote PE to detect transfer completion via a signal variable.

Args:

  * `dst`: CuTe tensor view pointing to the symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor.
  * `src`: CuTe tensor view pointing to the local source data on this PE.
  * `signal_var`: CuTe tensor view pointing to a symmetric signal variable (dtype `uint64`) on PE `pe`. Must be a 1-element NVSHMEM-allocated tensor.
  * `signal_val` (`int`): Value used to update the signal variable. Cast to `uint64`.
  * `signal_op`: Signal operation type. Supported values are `NVSHMEM_SIGNAL_SET` (set the signal to `signal_val`) and `NVSHMEM_SIGNAL_ADD` (atomically add `signal_val` to the signal variable).
  * `pe` (`int`): Target PE.

Note:
    The number of data elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a blocking operation: the data transfer and signal update complete before the call returns.

`nvshmem.core.device.cute.rma. put_signal_nbi ( dst , src , signal_var , signal_val , signal_op , pe )`

Non-blockingly puts data from `src` to symmetric `dst` on PE `pe`, then signals `signal_var`. This is a thread-level operation.

The signal operation atomically updates `signal_var` on the remote PE after the data transfer, allowing the remote PE to detect transfer completion via a signal variable.

Args:

  * `dst`: CuTe tensor view pointing to the symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor.
  * `src`: CuTe tensor view pointing to the local source data on this PE.
  * `signal_var`: CuTe tensor view pointing to a symmetric signal variable (dtype `uint64`) on PE `pe`. Must be a 1-element NVSHMEM-allocated tensor.
  * `signal_val` (`int`): Value used to update the signal variable. Cast to `uint64`.
  * `signal_op`: Signal operation type. Supported values are `NVSHMEM_SIGNAL_SET` (set the signal to `signal_val`) and `NVSHMEM_SIGNAL_ADD` (atomically add `signal_val` to the signal variable).
  * `pe` (`int`): Target PE.

Note:
    The number of data elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a non-blocking operation: neither the data transfer nor the signal update are guaranteed to be visible to the remote PE when the call returns.

`nvshmem.core.device.cute.rma. put_signal_warp ( dst , src , signal_var , signal_val , signal_op , pe )`

Puts data from `src` to symmetric `dst` on PE `pe`, then signals `signal_var`. This is a warp-level operation. All threads in the warp must call this function with the same arguments.

The signal operation atomically updates `signal_var` on the remote PE after the data transfer, allowing the remote PE to detect transfer completion via a signal variable.

Args:

  * `dst`: CuTe tensor view pointing to the symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor.
  * `src`: CuTe tensor view pointing to the local source data on this PE.
  * `signal_var`: CuTe tensor view pointing to a symmetric signal variable (dtype `uint64`) on PE `pe`. Must be a 1-element NVSHMEM-allocated tensor.
  * `signal_val` (`int`): Value used to update the signal variable. Cast to `uint64`.
  * `signal_op`: Signal operation type. Supported values are `NVSHMEM_SIGNAL_SET` (set the signal to `signal_val`) and `NVSHMEM_SIGNAL_ADD` (atomically add `signal_val` to the signal variable).
  * `pe` (`int`): Target PE.

Note:
    The number of data elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a blocking operation: the data transfer and signal update complete before the call returns.

`nvshmem.core.device.cute.rma. put_signal_nbi_block ( dst , src , signal_var , signal_val , signal_op , pe )`

Non-blockingly puts data from `src` to symmetric `dst` on PE `pe`, then signals `signal_var`. This is a CTA-level operation. All threads in the CTA must call this function with the same arguments.

The signal operation atomically updates `signal_var` on the remote PE after the data transfer, allowing the remote PE to detect transfer completion via a signal variable.

Args:

  * `dst`: CuTe tensor view pointing to the symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor.
  * `src`: CuTe tensor view pointing to the local source data on this PE.
  * `signal_var`: CuTe tensor view pointing to a symmetric signal variable (dtype `uint64`) on PE `pe`. Must be a 1-element NVSHMEM-allocated tensor.
  * `signal_val` (`int`): Value used to update the signal variable. Cast to `uint64`.
  * `signal_op`: Signal operation type. Supported values are `NVSHMEM_SIGNAL_SET` (set the signal to `signal_val`) and `NVSHMEM_SIGNAL_ADD` (atomically add `signal_val` to the signal variable).
  * `pe` (`int`): Target PE.

Note:
    The number of data elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a non-blocking operation: neither the data transfer nor the signal update are guaranteed to be visible to the remote PE when the call returns.

`nvshmem.core.device.cute.rma. put_signal_nbi_warp ( dst , src , signal_var , signal_val , signal_op , pe )`

Non-blockingly puts data from `src` to symmetric `dst` on PE `pe`, then signals `signal_var`. This is a warp-level operation. All threads in the warp must call this function with the same arguments.

The signal operation atomically updates `signal_var` on the remote PE after the data transfer, allowing the remote PE to detect transfer completion via a signal variable.

Args:

  * `dst`: CuTe tensor view pointing to the symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor.
  * `src`: CuTe tensor view pointing to the local source data on this PE.
  * `signal_var`: CuTe tensor view pointing to a symmetric signal variable (dtype `uint64`) on PE `pe`. Must be a 1-element NVSHMEM-allocated tensor.
  * `signal_val` (`int`): Value used to update the signal variable. Cast to `uint64`.
  * `signal_op`: Signal operation type. Supported values are `NVSHMEM_SIGNAL_SET` (set the signal to `signal_val`) and `NVSHMEM_SIGNAL_ADD` (atomically add `signal_val` to the signal variable).
  * `pe` (`int`): Target PE.

Note:
    The number of data elements transferred is `min(size(dst), size(src))`. `dst` and `src` must have the same element dtype. This is a non-blocking operation: neither the data transfer nor the signal update are guaranteed to be visible to the remote PE when the call returns.

# NVSHMEM4Py Memory Management with CuTe DSL

This section documents the NVSHMEM4Py Memory Management with CuTe DSL.

NVSHMEM4Py provides functions to access remote symmetric and multicast buffers as CuTe tensors via the `nvshmem.core.device.cute.mem` module.

`nvshmem.core.device.cute.mem. get_peer_tensor ( tensor: cutlass.cute.typing.Tensor , pe: cutlass.base_dsl.typing.Int32 )`

Returns a CuTe tensor view that aliases the symmetric tensor `tensor` on a remote PE `pe`.

Wraps `nvshmem_ptr` to translate the base pointer of `tensor` to the address of the corresponding symmetric allocation on PE `pe`, then reconstructs a tensor with the same layout over the remote memory region. The returned tensor can be used as a destination or source in NVSHMEM RMA operations without additional pointer arithmetic.

Args:

  * `tensor` (`cute.Tensor`): A CuTe tensor view backed by a symmetric (NVSHMEM-allocated) buffer on the calling PE. The layout of the returned tensor matches the layout of this argument.
  * `pe` (`cutlass.Int32`): Target PE whose symmetric copy of `tensor` is requested.

Returns:
    A `cute.Tensor` with the same dtype and layout as `tensor` but whose base pointer refers to the symmetric allocation on PE `pe`.
Note:
    The symmetric object must have been allocated with `nvshmem_malloc` (or equivalent) so that a corresponding allocation exists at the same symmetric offset on every PE. If PE `pe` is the calling PE this function returns a tensor equivalent to `tensor` itself.

`nvshmem.core.device.cute.mem. get_multicast_tensor ( team: cutlass.base_dsl.typing.Int32 , tensor: cutlass.cute.typing.Tensor )`

Returns a CuTe tensor view that aliases the symmetric tensor `tensor` via the multicast address for `team`.

Wraps `nvshmemx_mc_ptr` to obtain the multicast virtual address corresponding to the symmetric allocation backing `tensor` within `team`, then reconstructs a tensor with the same layout over that multicast memory region. Writes to the returned tensor are delivered to all PEs in `team` simultaneously, enabling efficient one-to-many communication patterns.

Args:

  * `team` (`cutlass.Int32`): NVSHMEM team handle identifying the set of PEs that share the multicast mapping. The calling PE must be a member of `team`.
  * `tensor` (`cute.Tensor`): A CuTe tensor view backed by a symmetric (NVSHMEM-allocated) buffer on the calling PE. The layout of the returned tensor matches the layout of this argument.

Returns:
    A `cute.Tensor` with the same dtype and layout as `tensor` but whose base pointer is the multicast virtual address for the symmetric allocation within `team`.
Note:
    Multicast support requires hardware and driver support (NVLink multicast or equivalent). The symmetric object must have been allocated with multicast support enabled. All PEs in `team` must participate in the multicast setup before any PE uses the returned tensor.
