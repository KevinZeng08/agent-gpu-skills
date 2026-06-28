# Memory Management

This section documents the memory management APIs in `nvshmem.core.memory`.

## Symmetric Memory in Python

In NVSHMEM4Py, symmetric memory must be explicitly managed due to Python’s garbage collection mechanism. Unlike in C/C++ where memory can be automatically freed when it goes out of scope, Python’s garbage collector may not immediately release memory resources, which can lead to resource leaks in distributed environments.

NVSHMEM4Py makes use of the NVIDIA `cuda.core` Python project’s memory interface to expose symmetric memory. Read CUDA.Core Memory docs. for more information.

NVSHMEM4Py uses `cuda.core.Buffer` objects to represent symmetric memory. These objects provide a DLPack-compatible interface that allows for seamless integration with other Python CUDA libraries like CuPy and PyTorch.

## Memory Lifecycle

The typical lifecycle of symmetric memory in NVSHMEM4Py is:

  1. **Allocation** : Call `nvshmem.core.buffer()` to allocate symmetric memory across all PEs.
  2. **Usage** : Use the returned `cuda.core.Buffer` object in your application.
  3. **Explicit Deallocation** : Call `nvshmem.core.free()` when the memory is no longer needed.

Even though Python has reference counting, it’s important to explicitly free NVSHMEM symmetric memory to ensure proper cleanup of distributed resources. Relying on Python’s garbage collection alone may lead to resource leaks or undefined behavior in a distributed environment.

Example:

    import nvshmem.core as nvshmem

    # Initialize NVSHMEM (initialization code not shown)

    # Allocate 1MB of symmetric memory
    sym_buffer = nvshmem.memory.buffer(1024 * 1024)

    # Use the buffer in your application
    # ...

    # Explicitly free the symmetric memory when done
    nvshmem.memory.free(sym_buffer)

    # Finalize NVSHMEM
    nvshmem.finalize()

## Memory API reference

The following functions relate to management of NVSHMEM symmetric memory in Python

`nvshmem.core.memory. buffer ( size , release=False , except_on_del=True ) → cuda.core._memory._buffer.Buffer`

Allocates an NVSHMEM-backed CUDA buffer.

Args:

size (int): The size in bytes of the buffer to allocate. release (bool, optional): Do not track this buffer internally to NVSHMEM

> If True, it is the user’s responsibility to hold references to the buffer until free() is called otherwise, deadlocks may occur.

except_on_del (bool, optional): If True, raise an exception if the buffer is not tracked or has already been freed.
    If False, print an error message.
Returns:
    `cuda.core.Buffer`: A DLPack-compatible CUDA buffer with NVSHMEM backing.
Raises:
    `NvshmemError`: If the buffer could not be allocated properly.

Note that this is a collective. All participating PEs must call `buffer()` in concert.

This operation runs on the cached Device. If the cached Device is not the current device, it will set the cached device to current and set it back at the end of the operation.

`nvshmem.core.memory. get_peer_buffer ( buffer: cuda.core._memory._buffer.Buffer , pe: int )`

Returns a peer buffer associated with an NVSHMEM-allocated object.

This is the Python object equivalent of nvshmem_ptr, which:

  * Given a pointer to an object on the NVSHMEM Symmetric Heap
  * Returns a pointer to a local object to which loads and stores can be performed

The Python equivalent returns a cuda.core.Buffer which starts at the address of the Buffer passed in, with the same size as the Buffer passed in.

For more information on nvshmem_ptr, see https://docs.nvidia.com/nvshmem/archives/nvshmem-101/api/docs/gen/api/setup.html#nvshmem-ptr

The get_peer_buffer function offers an efficient means to accomplish communication, for example when a sequence of reads and writes to a data object on a remote PE does not match the access pattern provided in other APIs.

Args:

  * buffer (`cuda.core.Buffer`): A buffer allocated with NVSHMEM.
  * pe (`int`): The peer’s PE

Returns:

  * `cuda.core.Buffer`: The buffer object representing the remote peer’s buffer.
    User need not call `nvshmem.core.free()` on this Buffer. It will be a no-op

Raises:

  * `NvshmemInvalid`: If the input buffer is not a valid NVSHMEM buffer.
  * `NvshmemError`: If the buffer is not tracked internally or no peer information is found.

`nvshmem.core.memory. free ( buffer: cuda.core._memory._buffer.Buffer ) → None`

Frees an NVSHMEM buffer that was previously allocated.

Args:
    buffer (`cuda.core.Buffer`): The buffer to free.
Raises:

  * `NvshmemInvalid`: If the buffer is not a valid NVSHMEM-managed buffer.
  * `NvshmemError`: If the buffer is not tracked or has already been freed.

Note that this is a collective. All participating PEs must call `free()` in concert.
