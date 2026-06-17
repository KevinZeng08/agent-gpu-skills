# Memory Registration Methods

Methods on [`Communicator`](class.md#nccl.core.Communicator) for registering buffers and windows for
zero-copy and RMA operations. The returned handle classes are documented
under [Memory Management](../memory.md).

## register_buffer

#### Communicator.register_buffer(buffer: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI) → [RegisteredBufferHandle](../resources.md#nccl.core.RegisteredBufferHandle)

Registers a buffer with this communicator for zero-copy communication.

Registered buffers can enable performance optimizations in NCCL
operations. Buffer size is automatically derived from buffer count
and dtype. The returned [`RegisteredBufferHandle`](../resources.md#nccl.core.RegisteredBufferHandle)
is tracked by the communicator and may be released explicitly via
its `close()` method, or
automatically when the communicator is destroyed or aborted.

* **Parameters:**
  **buffer** – Buffer to register (array, Buffer, or buffer-like
  object).
* **Returns:**
  [`RegisteredBufferHandle`](../resources.md#nccl.core.RegisteredBufferHandle) for the registered
  buffer.
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the buffer is on the wrong device or the
      communicator is not initialized.

#### SEE ALSO
[`ncclCommRegister()`](../../api/comms.md#c.ncclCommRegister)

## register_window

#### Communicator.register_window(buffer: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, flags: [WindowFlag](#nccl.core.WindowFlag) | None = None) → [RegisteredWindowHandle](../resources.md#nccl.core.RegisteredWindowHandle) | None

Collectively registers a local buffer into an NCCL window.

This is a collective call: every rank in the communicator must
participate, and buffer size must be equal among ranks by default.
Buffer size is automatically derived from buffer count and dtype.
If called within a group, the handle value may not be filled until
`ncclGroupEnd` completes. For non-blocking communicators, the handle
may remain `0` until [`get_async_error()`](status.md#nccl.core.Communicator.get_async_error) reports success.

The returned [`RegisteredWindowHandle`](../resources.md#nccl.core.RegisteredWindowHandle) is
tracked by the communicator and may be released explicitly via its
`close()` method, or
automatically when the communicator is destroyed or aborted.

* **Parameters:**
  * **buffer** – Local buffer to register as a window.
  * **flags** – Window registration flags. Defaults to `None`
    ([`DEFAULT`](#nccl.core.WindowFlag.DEFAULT)).
* **Returns:**
  [`RegisteredWindowHandle`](../resources.md#nccl.core.RegisteredWindowHandle) for the registered
  window, or `None` if NCCL returns a NULL handle (e.g. windows
  are unsupported on this platform).
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the buffer is on the wrong device or the
      communicator is not initialized.

#### SEE ALSO
[`ncclCommWindowRegister()`](../../api/comms.md#c.ncclCommWindowRegister)

## WindowFlag

### *class* nccl.core.WindowFlag(value, names=<not given>, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: `IntFlag`

Window registration behavior flags for
[`Communicator.register_window()`](#nccl.core.Communicator.register_window).

#### DEFAULT *= 0*

Default window registration.

#### COLL_SYMMETRIC *= 1*

Collective symmetric window registration.

#### STRICT_ORDERING *= 2*

Strict ordering for window operations.
