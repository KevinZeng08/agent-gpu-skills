# Remote Memory Access (RMA)

This section documents the Remote Memory Access (RMA) APIs in `nvshmem.core.rma`.

## RMA Operations in NVSHMEM4Py

NVSHMEM4Py provides a Pythonic interface to the RMA operations defined in the NVSHMEM specification. These operations allow for the transfer of data between processing elements (PEs) in a distributed environment.

NVSHMEM4Py supports two primary types of RMA operations:

  * **Put Operations** : Transfer data from the local PE to a remote PE
  * **Get Operations** : Retrieve data from a remote PE to the local PE

These operations follow the one-sided communication model, where the initiating PE specifies both the source and destination of the data transfer without requiring active participation from the remote PE.

## Supported RMA Operations

NVSHMEM4Py provides the following RMA operations:

  * **Put** : Copy data from a local buffer to a remote PE’s symmetric memory
  * **Get** : Copy data from a remote PE’s symmetric memory to a local buffer
  * **Put with Signal** : Copy data from a local buffer to a remote PE’s symmetric memory and signal a completion event to a separate remote PE.
  * **Wait on Signal** : Block execution until a signal is received from a remote PE indicating completion of a specified operation.

## Stream Requirement for RMA

Similar to collective operations, all RMA operations in NVSHMEM4Py require a CUDA stream to be provided. The stream is used for proper synchronization of GPU operations. See the CUDA.Core Stream docs for more details.

## Memory Management for RMA

NVSHMEM4Py requires explicit memory management. You must call `nvshmem.core.free()` on symmetric memory when you’re done with it. Relying on Python’s garbage collector will cause an `NvshmemError` exception, and the memory will be leaked until `nvshmem.core.finalize()` is called. This requirement exists to prevent deadlocks that could occur if the garbage collector attempted to free NVSHMEM symmetric memory.

## RMA Examples

**Put Example** :

    import nvshmem.core as nvshmem
    import cupy as cp
    from cuda.core.experimental import Device

    # Initialize NVSHMEM (initialization code not shown)

    # Get current device and create a stream
    device = Device()
    stream = device.create_stream()

    # Allocate symmetric memory
    size = 10
    src_array = nvshmem.array((size,), dtype=cp.float32)
    dest_array = nvshmem.array((size,), dtype=cp.float32)

    # Set values on local PE
    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    # Fill source array with PE ID
    src_array[:] = cp.ones(size, dtype=cp.float32) * my_pe

    # Target PE (circular next PE)
    target_pe = (my_pe + 1) % n_pes

    # Put data to the target PE
    nvshmem.put(dest_array, src_array, size, target_pe, stream=stream)

    # Ensure operation is complete
    stream.synchronize()

    # Clean up - explicit free is required
    nvshmem.free(src_array)
    nvshmem.free(dest_array)

**Get Example** :

    import nvshmem.core as nvshmem
    import cupy as cp
    from cuda.core.experimental import Device

    # Initialize NVSHMEM (initialization code not shown)

    # Get current device and create a stream
    device = Device()
    stream = device.create_stream()

    # Allocate symmetric memory
    size = 10
    src_array = nvshmem.array((size,), dtype=cp.float32)
    dest_array = nvshmem.array((size,), dtype=cp.float32)

    # Set values on each PE
    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    # Fill source array with PE ID
    src_array[:] = cp.ones(size, dtype=cp.float32) * my_pe

    # Target PE to get data from (circular previous PE)
    target_pe = (my_pe - 1 + n_pes) % n_pes

    # Get data from the target PE
    nvshmem.get(dest_array, src_array, size, target_pe, stream=stream)

    # Ensure operation is complete
    stream.synchronize()

    # Now dest_array contains data from the target PE

    # Clean up - explicit free is required
    nvshmem.free(src_array)
    nvshmem.free(dest_array)

**Put with Signal Example** :

    import nvshmem.core as nvshmem
    import cupy as cp
    from cuda.core.experimental import Device

    # Initialize NVSHMEM (initialization code not shown)

    # Get current device and create a stream
    device = Device()
    stream = device.create_stream()

    # Allocate symmetric memory
    size = 10
    src_array = nvshmem.array((size,), dtype=cp.float32)
    dest_array = nvshmem.array((size,), dtype=cp.float32)

    # Set values on each PE
    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    # Fill source array with PE ID
    src_array[:] = cp.ones(size, dtype=cp.float32) * my_pe

    # Target PE to get data from (circular previous PE)
    target_pe = (my_pe - 1 + n_pes) % n_pes
    # Source PE (the opposite of the target PE)
    src_pe = (my_pe + 1) % n_pes

    # A signal is always
    signal = nvshmem.core.array((1,), dtype="uint64")
    signal [:] = 0
    buf_sig, sz, type = nvshmem.core.array_get_buffer(signal)

    # Put with signal to the target PE
    nvshmem.core.put_signal(dest_array, src_array, buf_sig, 1, nvshmem.core.SignalOp.SIGNAL_SET, remote_pe=target_pe, stream=stream)

    # Wait on signal from the source PE
    nvshmem.core.signal_wait(buf_sig, 1, nvshmem.core.SignalOp.SIGNAL_SET, remote_pe=target_pe, stream=stream)

    # Ensure operation is complete
    stream.synchronize()

    # Now dest_array contains data from the target PE

    # Clean up - explicit free is required
    nvshmem.core.free_array(src_array)
    nvshmem.core.free_array(dest_array)

## RMA API reference

These functions are NVSHMEM4Py APIs that expose host-initiated remote memory accesses (RMA)

`nvshmem.core.rma. put_signal ( dst: object , src: object , signal_var: cuda.core._memory._buffer.Buffer , signal_val: int , signal_op: nvshmem.bindings.nvshmem.Signal_op , remote_pe: int = -1 , stream=None ) → None`

Performs a put with signal on a CUDA stream.

Args:

  * dst (object): Destination buffer (Buffer, Cupy array, or Torch tensor).
  * src (object): Source buffer (Buffer, Cupy array, or Torch tensor).
  * signal_var (Buffer): Symmetric memory buffer used as signal variable.
  * signal_val (int): Value to use in the signal operation.
  * signal_op (SignalOp): Signal operation type.
  * remote_pe (int): Target PE for the put.
  * stream (Stream): CUDA stream to issue the put on.

Raises:

  * `NotImplementedError`: If stream is None.
  * `ValueError`: If the signal buffer is invalid or too small.
  * `NvshmemError`: If any operations do not complete successfully

`nvshmem.core.rma. signal_op ( signal_var: cuda.core._memory._buffer.Buffer , signal_val: int , signal_op: nvshmem.bindings.nvshmem.Signal_op , remote_pe: int = -1 , stream: Union[None , nvshmem.core.nvshmem_types.NvshmemStream , cuda.core._stream.Stream] = None ) → None`

Performs a signal operation on a CUDA stream. Equivalent to a `put_signal` but with no data movement.

Args:

  * signal_var (Buffer): Symmetric memory buffer used as the signal variable.
  * signal_val (int): Value to use in the signal operation.
  * signal_op (SignalOp): Signal operation type.
  * remote_pe (int): Target PE for the signal operation.
  * stream (Stream): CUDA stream to issue the signal operation on.

Raises:

  * `NotImplementedError`: If stream is None.
  * `ValueError`: If the signal buffer is invalid or too small.
  * `NvshmemError`: If any operations do not complete successfully

`nvshmem.core.rma. signal_wait ( signal_var: cuda.core._memory._buffer.Buffer , signal_val: int , signal_op: nvshmem.bindings.nvshmem.Cmp_type , stream: Union[None , nvshmem.core.nvshmem_types.NvshmemStream , cuda.core._stream.Stream] = None ) → None`

Waits until a symmetric signal variable satisfies a given condition.

Args:

  * signal_var (Buffer): Symmetric memory buffer used as the signal source.
  * signal_val (int): Value to compare against.
  * signal_op (SignalOp): Wait condition
  * stream (Stream): CUDA stream to issue the wait on.

Raises:

  * `NotImplementedError`: If stream is None.
  * `ValueError`: If the signal buffer is invalid.
  * `NvshmemError`: If any operations do not complete successfully

`nvshmem.core.rma. put ( dst: object , src: object , remote_pe: int = -1 , stream: Union[None , nvshmem.core.nvshmem_types.NvshmemStream , cuda.core._stream.Stream] = None )`

Performs a host-initiated NVSHMEM put operation on a CUDA stream.

Args:

  * dst (object): Destination buffer (Buffer, Cupy array, or Torch tensor).
  * src (object): Source buffer (Buffer, Cupy array, or Torch tensor).
  * remote_pe (int): Target PE for the put.
  * stream (Stream): CUDA stream to issue the put on.

Raises:

  * `NotImplementedError`: If stream is `None`.
  * `NvshmemInvalid`: If inputs are not valid Buffer-compatible types.
  * `NvshmemError`: If any operations do not complete successfully

`nvshmem.core.rma. get ( dst: object , src: object , remote_pe: int = -1 , stream: Union[None , nvshmem.core.nvshmem_types.NvshmemStream , cuda.core._stream.Stream] = None )`

Performs a host-initiated NVSHMEM get operation on a CUDA stream.

Args:

  * dst (object): Destination buffer (Buffer, Cupy array, or Torch tensor).
  * src (object): Source buffer (Buffer, Cupy array, or Torch tensor).
  * remote_pe (int): Target PE for the get.
  * stream (Stream): CUDA stream to issue the get on.

Raises:

  * `NotImplementedError`: If stream is None.
  * `ValueError`: If inputs are not valid Buffer-compatible types.
  * `NvshmemError`: If any operations do not complete successfully

`nvshmem.core.rma. quiet ( stream: Union[None , nvshmem.core.nvshmem_types.NvshmemStream , cuda.core._stream.Stream] = None ) → None`

Ensures completion of all previously issued NVSHMEM operations on the given stream.

This is equivalent to a device-side `nvshmem_quiet` for host-initiated NVSHMEM operations.

Args:

  * stream (`Stream`): CUDA stream to synchronize.

Raises:

  * `NotImplementedError`: If stream is None.
  * `NvshmemError`: If any operations do not complete successfully

`nvshmem.core.rma.``SignalOp`

alias of `nvshmem.bindings.nvshmem.Signal_op`

`class nvshmem.core. SignalOp ( IntEnum )`

This is an enum representing the available Signal operations to to be used with `put_signal` and `signal_wait`.

> `SIGNAL_SET`
>
>
> When this operation is used, the NVSHMEM library will set the signal to the value provided
>
> `SIGNAL_ADD`
>
>
> When this operation is used, the NVSHMEM library will add the value stored in the signal to the value provided
