# NVSHMEM Device Atomic Memory Operations with CuTe DSL

This section documents the NVSHMEM Device Atomic Memory Operations with CuTe DSL.

## Example: Using `atomic_add` in a CuTe kernel

The following example demonstrates how to use the NVSHMEM atomic add operation (`atomic_add`) in a CuTe kernel. This allows threads to safely increment a value in shared memory on a remote PE (processing element).

    import cutlass
    from cutlass import cute
    from cuda.core import Device, Stream
    import nvshmem
    import nvshmem.core.device.cute as nvshmem_cute
    import nvshmem.core.interop.cute as nvshmem_cute_interop
    from mpi4py import MPI

    @cute.kernel
    def atomic_add_kernel(dst: cute.Tensor, value: cutlass.Int32, pe: cutlass.Int32):
        # Atomically add 'value' to dst on remote PE
        nvshmem_cute.atomic_add(dst, value, pe)

    @cute.jit
    def atomic_add_launcher(dst, value, pe):
        atomic_add_kernel[1, 1](dst, value, pe)

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

    # Allocate device tensor using NVSHMEM symmetric memory
    dst = nvshmem_cute_interop.tensor((1,), dtype=cute.Int32)

    # Compile and launch the kernel
    compiled_fn, nvshmem_kernel = nvshmem_cute_interop.cute_compile_helper(
        atomic_add_launcher, dst, 42, pe
    )
    compiled_fn(dst, 42, pe, stream=stream)

    # Finalize NVSHMEM
    nvshmem.core.library_finalize(nvshmem_kernel)
    nvshmem_cute_interop.cleanup_cute()
    nvshmem.finalize(dev=dev, stream=stream)

This example atomically adds 42 to the `dst` tensor on the next PE in a ring. The atomic operation is performed at the thread level, ensuring safe concurrent access.

`nvshmem.core.device.cute.amo. atomic_inc ( dst , pe )`

Atomically increments the value at symmetric `dst` on PE `pe` by 1. Does not return the old value.

This is a thread-level remote atomic increment operation (non-fetching variant). Use `atomic_fetch_inc` if you need the value before the increment.

Args:

  * `dst`: CuTe tensor view pointing to a single-element symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor. Supported dtypes are integral types as determined by the NVSHMEM atomic inc dispatch table.
  * `pe` (`int`): Target PE.

Note:
    Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.amo. atomic_fetch_inc ( dst , pe )`

Atomically increments the value at symmetric `dst` on PE `pe` by 1, and returns the value prior to the increment.

This is a thread-level remote atomic fetch-and-increment operation.

Args:

  * `dst`: CuTe tensor view pointing to a single-element symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor. Supported dtypes are integral types as determined by the NVSHMEM atomic fetch-inc dispatch table.
  * `pe` (`int`): Target PE.

Returns:
    The value stored at `dst` on PE `pe` prior to the increment.
Note:
    Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.amo. atomic_fetch ( src , pe )`

Atomically fetches (reads) the current value at symmetric `src` on PE `pe`.

This is a thread-level remote atomic operation. The read is performed atomically with respect to other atomic operations on the same location.

Args:

  * `src`: CuTe tensor view pointing to a single-element symmetric source on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor. The element dtype determines which underlying NVSHMEM atomic is dispatched.
  * `pe` (`int`): Source PE to fetch from.

Returns:
    The current value stored at `src` on PE `pe`, with the same dtype as `src`.
Note:
    Supported dtypes are determined by the NVSHMEM atomic fetch dispatch table. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.amo. atomic_set ( dst , value , pe )`

Atomically sets the value at symmetric `dst` on PE `pe` to `value`.

This is a thread-level remote atomic store. The write is performed atomically with respect to other atomic operations on the same location.

Args:

  * `dst`: CuTe tensor view pointing to a single-element symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor. The element dtype determines which underlying NVSHMEM atomic is dispatched.
  * `value`: The value to store. Cast to the element dtype of `dst` before the operation.
  * `pe` (`int`): Target PE.

Note:
    This operation does not return the old value. Use `atomic_fetch` before setting if you need the previous value. Supported dtypes are determined by the NVSHMEM atomic set dispatch table. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.amo. atomic_add ( dst , value , pe )`

Atomically adds `value` to the value at symmetric `dst` on PE `pe`. Does not return the old value.

This is a thread-level remote atomic add operation (non-fetching variant). Use `atomic_fetch_add` if you need the value before the addition.

Args:

  * `dst`: CuTe tensor view pointing to a single-element symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor. The element dtype determines which underlying NVSHMEM atomic is dispatched (integral and floating-point types supported).
  * `value`: Value to add. Cast to the element dtype of `dst`.
  * `pe` (`int`): Target PE.

Note:
    Supported dtypes are determined by the NVSHMEM atomic add dispatch table. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.amo. atomic_fetch_add ( dst , value , pe )`

Atomically adds `value` to the value at symmetric `dst` on PE `pe`, and returns the value prior to the addition.

This is a thread-level remote atomic fetch-and-add operation.

Args:

  * `dst`: CuTe tensor view pointing to a single-element symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor. The element dtype determines which underlying NVSHMEM atomic is dispatched (integral and floating-point types supported).
  * `value`: Value to add. Cast to the element dtype of `dst`.
  * `pe` (`int`): Target PE.

Returns:
    The value stored at `dst` on PE `pe` prior to the addition.
Note:
    Supported dtypes are determined by the NVSHMEM atomic fetch-add dispatch table. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.amo. atomic_and ( dst , value , pe )`

Atomically applies bitwise AND of `value` with the value at symmetric `dst` on PE `pe`. Does not return the old value.

This is a thread-level remote atomic bitwise AND operation (non-fetching variant). Use `atomic_fetch_and` if you need the value before the operation.

Args:

  * `dst`: CuTe tensor view pointing to a single-element symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor. Only integral (bitwise) dtypes are supported.
  * `value`: Mask value to AND with. Cast to the element dtype of `dst`.
  * `pe` (`int`): Target PE.

Note:
    Only integral dtypes (e.g., `uint32`, `uint64`, etc.) are supported. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.amo. atomic_fetch_and ( dst , value , pe )`

Atomically applies bitwise AND of `value` with the value at symmetric `dst` on PE `pe`, and returns the value prior to the operation.

This is a thread-level remote atomic fetch-and-AND operation.

Args:

  * `dst`: CuTe tensor view pointing to a single-element symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor. Only integral (bitwise) dtypes are supported.
  * `value`: Mask value to AND with. Cast to the element dtype of `dst`.
  * `pe` (`int`): Target PE.

Returns:
    The value stored at `dst` on PE `pe` prior to the AND operation.
Note:
    Only integral dtypes (e.g., `uint32`, `uint64`, etc.) are supported. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.amo. atomic_or ( dst , value , pe )`

Atomically applies bitwise OR of `value` with the value at symmetric `dst` on PE `pe`. Does not return the old value.

This is a thread-level remote atomic bitwise OR operation (non-fetching variant). Use `atomic_fetch_or` if you need the value before the operation.

Args:

  * `dst`: CuTe tensor view pointing to a single-element symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor. Only integral (bitwise) dtypes are supported.
  * `value`: Mask value to OR with. Cast to the element dtype of `dst`.
  * `pe` (`int`): Target PE.

Note:
    Only integral dtypes (e.g., `uint32`, `uint64`, etc.) are supported. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.amo. atomic_fetch_or ( dst , value , pe )`

Atomically applies bitwise OR of `value` with the value at symmetric `dst` on PE `pe`, and returns the value prior to the operation.

This is a thread-level remote atomic fetch-and-OR operation.

Args:

  * `dst`: CuTe tensor view pointing to a single-element symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor. Only integral (bitwise) dtypes are supported.
  * `value`: Mask value to OR with. Cast to the element dtype of `dst`.
  * `pe` (`int`): Target PE.

Returns:
    The value stored at `dst` on PE `pe` prior to the OR operation.
Note:
    Only integral dtypes (e.g., `uint32`, `uint64`, etc.) are supported. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.amo. atomic_xor ( dst , value , pe )`

Atomically applies bitwise XOR of `value` with the value at symmetric `dst` on PE `pe`. Does not return the old value.

This is a thread-level remote atomic bitwise XOR operation (non-fetching variant). Use `atomic_fetch_xor` if you need the value before the operation.

Args:

  * `dst`: CuTe tensor view pointing to a single-element symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor. Only integral (bitwise) dtypes are supported.
  * `value`: Mask value to XOR with. Cast to the element dtype of `dst`.
  * `pe` (`int`): Target PE.

Note:
    Only integral dtypes (e.g., `uint32`, `uint64`, etc.) are supported. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.amo. atomic_fetch_xor ( dst , value , pe )`

Atomically applies bitwise XOR of `value` with the value at symmetric `dst` on PE `pe`, and returns the value prior to the operation.

This is a thread-level remote atomic fetch-and-XOR operation.

Args:

  * `dst`: CuTe tensor view pointing to a single-element symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor. Only integral (bitwise) dtypes are supported.
  * `value`: Mask value to XOR with. Cast to the element dtype of `dst`.
  * `pe` (`int`): Target PE.

Returns:
    The value stored at `dst` on PE `pe` prior to the XOR operation.
Note:
    Only integral dtypes (e.g., `uint32`, `uint64`, etc.) are supported. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.amo. atomic_swap ( dst , value , pe )`

Atomically replaces the value at symmetric `dst` on PE `pe` with `value`, and returns the old value.

This is a thread-level remote atomic swap operation.

Args:

  * `dst`: CuTe tensor view pointing to a single-element symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor. The element dtype determines which underlying NVSHMEM atomic is dispatched.
  * `value`: New value to store. Cast to the element dtype of `dst`.
  * `pe` (`int`): Target PE.

Returns:
    The value stored at `dst` on PE `pe` prior to the swap.
Note:
    Supported dtypes are determined by the NVSHMEM atomic swap dispatch table. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.amo. atomic_compare_swap ( dst , cond , value , pe )`

Atomically compares the value at symmetric `dst` on PE `pe` with `cond`, and if equal, replaces it with `value`. Returns the old value regardless.

This is a thread-level remote atomic compare-and-swap (CAS) operation.

Args:

  * `dst`: CuTe tensor view pointing to a single-element symmetric destination on PE `pe`. Must be a symmetric (NVSHMEM-allocated) tensor. The element dtype determines which underlying NVSHMEM atomic is dispatched.
  * `cond`: Comparison value. Cast to the element dtype of `dst`. The swap only occurs if the current value at `dst` equals `cond`.
  * `value`: Replacement value. Cast to the element dtype of `dst`. Written to `dst` only if the comparison succeeds.
  * `pe` (`int`): Target PE.

Returns:
    The value stored at `dst` on PE `pe` prior to the operation, regardless of whether the swap occurred.
Note:
    Supported dtypes are determined by the NVSHMEM atomic compare-swap dispatch table. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.
