# Point-to-Point and Signal Methods

Methods on [`Communicator`](class.md#nccl.core.Communicator) for point-to-point and signal/wait
operations. See [Point To Point Communication Functions](../../api/p2p.md) for the corresponding C API.

## send

#### Communicator.send(sendbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, peer: int, \*, stream: [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream) | cuda.core.typing.IsStreamType | int | None = None) → None

Sends a buffer to a peer rank.

* **Parameters:**
  * **sendbuf** – Source buffer to send.
  * **peer** – Destination rank ID.
  * **stream** – CUDA stream for the operation. Defaults to `None` (the
    default stream).
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the buffer specification is invalid, the buffer
      is on the wrong device, or the communicator is not
      initialized.

#### SEE ALSO
[`ncclSend()`](../../api/p2p.md#c.ncclSend)

## recv

#### Communicator.recv(recvbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, peer: int, \*, stream: [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream) | cuda.core.typing.IsStreamType | int | None = None) → None

Receives data into a buffer from a peer rank.

* **Parameters:**
  * **recvbuf** – Destination buffer to receive into.
  * **peer** – Source rank ID.
  * **stream** – CUDA stream for the operation. Defaults to `None` (the
    default stream).
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the buffer specification is invalid, the buffer
      is on the wrong device, or the communicator is not
      initialized.

#### SEE ALSO
[`ncclRecv()`](../../api/p2p.md#c.ncclRecv)

## signal

#### Communicator.signal(peer: int, signal_index: int = 0, context: int = 0, flags: int = 0, \*, stream: [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream) | cuda.core.typing.IsStreamType | int | None = None) → None

Sends a signal to a peer rank.

Enqueues a signal operation on the specified CUDA stream that
notifies the target peer rank. The peer can wait for this signal
using [`wait_signal()`](#nccl.core.Communicator.wait_signal).

* **Parameters:**
  * **peer** – Target rank to send the signal to.
  * **signal_index** – Signal index identifier. Currently must be 0.
  * **context** – Context identifier. Currently must be 0.
  * **flags** – Reserved for future use. Currently must be 0.
  * **stream** – CUDA stream to enqueue the signal operation on. Defaults
    to `None` (the default stream).
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

#### SEE ALSO
[`ncclSignal()`](../../api/p2p.md#c.ncclSignal)

## wait_signal

#### Communicator.wait_signal(descs: [WaitSignalDesc](#nccl.core.WaitSignalDesc) | Sequence[[WaitSignalDesc](#nccl.core.WaitSignalDesc)], \*, stream: [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream) | cuda.core.typing.IsStreamType | int | None = None) → None

Waits for signals as described by the signal descriptor(s).

Enqueues a wait operation on the specified CUDA stream that blocks
until the required signals from peer ranks are received. Each
descriptor specifies a peer rank and the number of signal
operations to wait for from that peer.

* **Parameters:**
  * **descs** – One or more [`WaitSignalDesc`](#nccl.core.WaitSignalDesc) descriptors
    specifying which peers to wait for and how many signals to
    expect from each.
  * **stream** – CUDA stream to enqueue the wait operation on. Defaults
    to `None` (the default stream).
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

#### SEE ALSO
[`ncclWaitSignal()`](../../api/p2p.md#c.ncclWaitSignal)

## put_signal

#### Communicator.put_signal(local_buffer: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, peer: int, peer_window: [RegisteredWindowHandle](../resources.md#nccl.core.RegisteredWindowHandle), peer_window_offset: int = 0, signal_index: int = 0, context: int = 0, flags: int = 0, \*, stream: [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream) | cuda.core.typing.IsStreamType | int | None = None) → None

Puts data from a local buffer to a peer’s window and sends a signal.

Enqueues a put-with-signal operation on the specified CUDA stream
that transfers the local buffer contents to the target peer’s
registered window and notifies that peer. The peer can wait for
this signal (and thus for the put to complete) using
[`wait_signal()`](#nccl.core.Communicator.wait_signal). The peer’s memory must be registered with
[`register_window()`](registration.md#nccl.core.Communicator.register_window); pass the peer’s window handle as
`peer_window` (e.g. obtained via an allgather of window handles).

* **Parameters:**
  * **local_buffer** – Source buffer whose contents are put to the peer.
  * **peer** – Target rank to put the data to and send the signal to.
  * **peer_window** – Peer’s [`RegisteredWindowHandle`](../resources.md#nccl.core.RegisteredWindowHandle)
    (from [`register_window()`](registration.md#nccl.core.Communicator.register_window)).
  * **peer_window_offset** – Offset in the peer’s window in elements.
    Defaults to 0.
  * **signal_index** – Signal index identifier. Currently must be 0.
  * **context** – Context identifier. Currently must be 0.
  * **flags** – Reserved for future use. Currently must be 0.
  * **stream** – CUDA stream to enqueue the put_signal operation on.
    Defaults to `None` (the default stream).
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized, or if the
      buffer specification is invalid or the buffer is on a
      different device than the communicator.

#### SEE ALSO
[`ncclPutSignal()`](../../api/p2p.md#c.ncclPutSignal)

## WaitSignalDesc

### *class* nccl.core.WaitSignalDesc(peer: int, op_count: int = 1, signal_index: int = 0, context: int = 0)

Bases: `object`

Descriptor for a wait-signal operation.

Describes a single signal-wait operation for use with
[`Communicator.wait_signal()`](#nccl.core.Communicator.wait_signal). Each descriptor specifies which peer
to wait for, how many signal operations to wait for, and additional
context for the wait operation.

#### peer *: int*

Target peer rank to wait for signals from.

#### op_count *: int* *= 1*

Number of signal operations to wait for from the peer. Defaults to 1.

#### signal_index *: int* *= 0*

Signal index identifier. Currently must be 0. Defaults to 0.

#### context *: int* *= 0*

Context identifier. Currently must be 0. Defaults to 0.
