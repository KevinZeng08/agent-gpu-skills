# Communicator Resources

Resource handles owned by a [`Communicator`](communicator/class.md#nccl.core.Communicator). They share a common
lifecycle: each handle is tracked by its owning communicator and is
released either explicitly via its `close()` method or automatically
when the communicator is destroyed or aborted.

## CommResource

### *class* nccl.core.resources.CommResource(comm_ptr: int)

Bases: `ABC`

Abstract base class for NCCL communicator-owned resources.

Resources are tied to a specific communicator. They can be released
explicitly via [`close()`](#nccl.core.resources.CommResource.close), and are released automatically when the
owning communicator is destroyed or aborted.

#### close() → None

Explicitly deallocates the resource.

Idempotent: safe to call multiple times.

#### *property* is_valid *: bool*

Whether the resource has been initialized and is still valid (not closed).

## RegisteredBufferHandle

### *class* nccl.core.RegisteredBufferHandle(comm_ptr: int, buffer_ptr: int, size: int)

Bases: [`CommResource`](#nccl.core.resources.CommResource)

NCCL registered buffer handle for zero-copy optimized communication.

Registers a user buffer with the communicator to enable performance
optimizations in NCCL operations. Created by
[`Communicator.register_buffer()`](communicator/registration.md#nccl.core.Communicator.register_buffer). The registration handle can be
released explicitly via `close()`, or automatically when the
owning communicator is destroyed or aborted.

#### *property* handle *: int*

Registration handle for NCCL operations.

* **Raises:**
  **RuntimeError** – If the buffer has been deregistered or the handle
      is invalid.

#### *property* size *: int*

Size of the registered buffer in bytes.

## RegisteredWindowHandle

### *class* nccl.core.RegisteredWindowHandle(comm_ptr: int, buffer_ptr: int, size: int, flags: [WindowFlag](communicator/registration.md#nccl.core.WindowFlag) | None = None)

Bases: [`CommResource`](#nccl.core.resources.CommResource)

NCCL registered window handle for Remote Memory Access (RMA) operations.

Registers a memory window with the communicator for one-sided communication
patterns. Created by [`Communicator.register_window()`](communicator/registration.md#nccl.core.Communicator.register_window). Registration
is collective: all ranks must call
[`Communicator.register_window()`](communicator/registration.md#nccl.core.Communicator.register_window) with equal buffer sizes by
default. Deregistration is local. The window handle can be released
explicitly via `close()`, or automatically when the owning
communicator is destroyed or aborted.

#### *property* is_valid *: bool*

Whether the resource has been initialized and is still valid (not closed).

#### *property* handle *: int*

Window handle for NCCL operations.

* **Raises:**
  **RuntimeError** – If the window has been deregistered or the handle
      is invalid.

#### *property* size *: int*

Size of the registered window in bytes.

#### *property* user_ptr *: int*

Original user buffer pointer registered with this window.

* **Raises:**
  **RuntimeError** – If the window has been deregistered.

#### get_lsa_multimem_device_pointer(offset: int = 0) → int | None

Returns the LSA multicast device pointer for this window.

Returns a device pointer suitable for multicast operations over the
LSA (Load/Store Accessible) team. The pointer is valid as long as
the window and communicator remain alive.

* **Parameters:**
  **offset** – Byte offset within the window buffer. Defaults to 0.
* **Returns:**
  Device pointer as int, or `None` if multimem is not supported.
* **Raises:**
  **RuntimeError** – If the window has been closed.

#### get_lsa_device_pointer(lsa_rank: int, offset: int = 0) → int

Returns the LSA device pointer for a peer within the LSA team.

Returns a device pointer to the peer’s window buffer addressable
from the local GPU via LSA (Load/Store Accessible) mapping.

* **Parameters:**
  * **lsa_rank** – Rank within the LSA team (0 to lsa_size - 1).
  * **offset** – Byte offset within the window buffer. Defaults to 0.
* **Returns:**
  Device pointer as int.
* **Raises:**
  **RuntimeError** – If the window has been closed.

#### get_peer_device_pointer(peer: int, offset: int = 0) → int | None

Returns a device pointer to a peer’s window buffer by world rank.

If the peer is not reachable via LSA, returns `None`.

* **Parameters:**
  * **peer** – World rank of the peer (0 to nranks - 1).
  * **offset** – Byte offset within the window buffer. Defaults to 0.
* **Returns:**
  Device pointer as int, or `None` if the peer is not reachable
  via LSA.
* **Raises:**
  **RuntimeError** – If the window has been closed.

## CustomRedOp

### *class* nccl.core.CustomRedOp(comm_ptr: int, scalar_ptr: int, datatype: [NcclDataType](types.md#nccl.core.NcclDataType), residence: nccl.bindings.nccl.ScalarResidence)

Bases: [`CommResource`](#nccl.core.resources.CommResource)

NCCL user-defined custom reduction operator.

Created by [`Communicator.create_pre_mul_sum()`](communicator/collectives.md#nccl.core.Communicator.create_pre_mul_sum). The PreMulSum
operator performs `output = scalar * sum(inputs)`, useful for averaging
or weighted reductions. The operator can be released explicitly via
`close()`, or automatically when the owning communicator is
destroyed or aborted.

#### *property* op *: int*

Operator handle for use in reduction operations.

* **Raises:**
  **RuntimeError** – If the operator has been destroyed or is invalid.

## DevCommResource

### *class* nccl.core.DevCommResource(comm_ptr: int, requirements_ptr: int)

Bases: [`CommResource`](#nccl.core.resources.CommResource)

NCCL device communicator resource for device-side operations.

Wraps `ncclDevComm_t` and manages its lifecycle. Created by
[`Communicator.create_dev_comm()`](communicator/device_setup.md#nccl.core.Communicator.create_dev_comm). The device communicator is
automatically destroyed when the parent communicator is destroyed or
aborted.

#### *property* ptr *: int*

Raw pointer to the underlying [`ncclDevComm_t`](../api/device_setup.md#c.ncclDevComm) structure.

* **Raises:**
  **RuntimeError** – If the device communicator has been destroyed.
