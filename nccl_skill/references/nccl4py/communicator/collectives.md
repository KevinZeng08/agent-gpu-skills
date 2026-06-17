# Collective Communication Methods

Methods on [`Communicator`](class.md#nccl.core.Communicator) for collective communication. See
[Collective Communication Functions](../../api/colls.md) for the corresponding C API.

## allreduce

#### Communicator.allreduce(sendbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, recvbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, op: [NcclRedOp](../types.md#nccl.core.NcclRedOp) | [CustomRedOp](../resources.md#nccl.core.CustomRedOp), \*, stream: [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream) | cuda.core.typing.IsStreamType | int | None = None) → None

All-reduce variant of [`reduce()`](#nccl.core.Communicator.reduce).

Equivalent to `reduce(sendbuf, recvbuf, op, root=None, stream=stream)`:
reduces data across all ranks and stores identical copies in each
rank’s recvbuf. See [`reduce()`](#nccl.core.Communicator.reduce) for argument semantics.

#### SEE ALSO
[`reduce()`](#nccl.core.Communicator.reduce), [`ncclAllReduce()`](../../api/colls.md#c.ncclAllReduce)

## broadcast

#### Communicator.broadcast(sendbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI | Any, recvbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, root: int, \*, stream: [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream) | cuda.core.typing.IsStreamType | int | None = None) → None

Copies data from `sendbuf` on the root rank to all ranks’ `recvbuf`.

`sendbuf` is only used on the root rank and is ignored on other
ranks.

On the root rank, both buffers must have matching data types and
`sendcount == recvcount`. Element count is inferred from
`recvbuf`: `count = recvcount`. In-place operation occurs when
`sendbuf` and `recvbuf` resolve to the same device memory
address.

* **Parameters:**
  * **sendbuf** – Source buffer (only used on the root rank).
  * **recvbuf** – Destination buffer that will receive the broadcast data.
  * **root** – Root rank that broadcasts the data (0 to `nranks - 1`).
  * **stream** – CUDA stream for the operation. Defaults to `None` (the
    default stream).
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If send and receive buffers have mismatched dtypes,
      mismatched counts, are on the wrong device, are invalid
      specifications, or the communicator is not initialized.

#### SEE ALSO
[`ncclBroadcast()`](../../api/colls.md#c.ncclBroadcast)

## reduce

#### Communicator.reduce(sendbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, recvbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI | Any, op: [NcclRedOp](../types.md#nccl.core.NcclRedOp) | [CustomRedOp](../resources.md#nccl.core.CustomRedOp), root: int | None = None, \*, stream: [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream) | cuda.core.typing.IsStreamType | int | None = None) → None

Reduces data from all ranks using the specified operation.

Supports two modes. In AllReduce mode (`root` is `None`) all
ranks receive the reduced result in `recvbuf`. In Reduce mode
(`root` specified) only the root rank receives the reduced
result; `recvbuf` is ignored on other ranks.

Both buffers must have matching data types where used. Element
count is inferred from `sendbuf`: `count = sendcount`. In
AllReduce mode, all ranks must have `recvcount >= sendcount`;
in Reduce mode, only the root rank requires
`recvcount >= sendcount`. In-place operation occurs when
`sendbuf` and `recvbuf` resolve to the same device memory
address.

* **Parameters:**
  * **sendbuf** – Source buffer containing data to be reduced.
  * **recvbuf** – Destination buffer for the reduced result. Only used on
    the root rank in Reduce mode.
  * **op** – Reduction operator (e.g. `SUM`,
    `MAX`, `MIN`,
    `AVG`, `PROD`, or a
    [`CustomRedOp`](../resources.md#nccl.core.CustomRedOp)).
  * **root** – Root rank that receives the reduced result (0 to
    `nranks - 1`). If `None`, performs an all-reduce.
    Defaults to `None`.
  * **stream** – CUDA stream for the operation. Defaults to `None` (the
    default stream).
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If send and receive buffers have mismatched dtypes,
      mismatched counts, are on the wrong device, are invalid
      specifications, or the communicator is not initialized.

#### SEE ALSO
[`ncclAllReduce()`](../../api/colls.md#c.ncclAllReduce), [`ncclReduce()`](../../api/colls.md#c.ncclReduce)

## allgather

#### Communicator.allgather(sendbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, recvbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, \*, stream: [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream) | cuda.core.typing.IsStreamType | int | None = None) → None

All-gather variant of [`gather()`](#nccl.core.Communicator.gather).

Equivalent to `gather(sendbuf, recvbuf, root=None, stream=stream)`:
gathers `sendcount` values from each rank and places identical
copies of the concatenated result in every rank’s recvbuf. See
[`gather()`](#nccl.core.Communicator.gather) for argument semantics.

#### SEE ALSO
[`gather()`](#nccl.core.Communicator.gather), [`ncclAllGather()`](../../api/colls.md#c.ncclAllGather)

## reduce_scatter

#### Communicator.reduce_scatter(sendbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, recvbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, op: [NcclRedOp](../types.md#nccl.core.NcclRedOp) | [CustomRedOp](../resources.md#nccl.core.CustomRedOp), \*, stream: [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream) | cuda.core.typing.IsStreamType | int | None = None) → None

Reduces data from all ranks and scatters the result across ranks.

Each rank receives a different portion of the reduced result: rank
`i` receives the i-th block in its `recvbuf`.

Both buffers must have matching data types. Element count is
inferred from `sendbuf`: `count = sendcount / nranks`.
`sendcount` must be `>= nranks` and `recvcount` must be
`>= count`. In-place operation occurs when `recvbuf` resolves
to `sendbuf_address + rank * count`.

* **Parameters:**
  * **sendbuf** – Source buffer (size `>= nranks * recvcount` elements).
  * **recvbuf** – Destination buffer with `recvcount` elements.
  * **op** – Reduction operator (e.g. `SUM`,
    `MAX`, `MIN`,
    `AVG`, `PROD`, or a
    [`CustomRedOp`](../resources.md#nccl.core.CustomRedOp)).
  * **stream** – CUDA stream for the operation. Defaults to `None` (the
    default stream).
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If send and receive buffers have mismatched dtypes,
      `sendbuf` is too small, are on the wrong device, are
      invalid specifications, or the communicator is not
      initialized.

#### SEE ALSO
[`ncclReduceScatter()`](../../api/colls.md#c.ncclReduceScatter)

## alltoall

#### Communicator.alltoall(sendbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, recvbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, \*, stream: [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream) | cuda.core.typing.IsStreamType | int | None = None) → None

Each rank sends and receives `count` values to and from every other rank.

Data sent to destination rank `j` is taken from
`sendbuf + j * count` and data received from source rank `i` is
placed at `recvbuf + i * count`.

Both buffers must have matching data types. Element count is
inferred from `sendbuf`: `count = sendcount / nranks`.
`sendcount` must be `>= nranks` and `recvcount` must be
`>= sendcount`.

* **Parameters:**
  * **sendbuf** – Source buffer (size `>= nranks * count` elements).
  * **recvbuf** – Destination buffer (size `>= nranks * count` elements).
  * **stream** – CUDA stream for the operation. Defaults to `None` (the
    default stream).
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If send and receive buffers have mismatched dtypes,
      buffer sizes are incompatible with `nranks`, are on the
      wrong device, are invalid specifications, or the
      communicator is not initialized.

#### SEE ALSO
[`ncclAlltoAll()`](../../api/colls.md#c.ncclAlltoAll)

## gather

#### Communicator.gather(sendbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, recvbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI | Any, root: int | None = None, \*, stream: [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream) | cuda.core.typing.IsStreamType | int | None = None) → None

Gathers `sendcount` values from all ranks.

Supports two modes. In AllGather mode (`root` is `None`) values
are gathered from all ranks and identical copies of the result are
placed in each `recvbuf`. In Gather mode (`root` specified)
values are gathered to the specified root rank only; `recvbuf`
is ignored on other ranks.

Both buffers must have matching data types where used. Element
count is inferred from `sendbuf`: `count = sendcount`. Data
from rank `i` is placed at `recvbuf + i * sendcount`. AllGather
mode requires `recvcount >= nranks * sendcount` on every rank;
Gather mode requires it only on the root rank.

In-place operation occurs when `sendbuf` resolves to
`recvbuf_address + rank * sendcount` in AllGather mode, or to
`recvbuf_address + root * sendcount` in Gather mode.

* **Parameters:**
  * **sendbuf** – Source buffer containing `sendcount` elements.
  * **recvbuf** – Destination buffer (size `>= nranks * sendcount`
    elements). In Gather mode, only used on the root rank.
  * **root** – Root rank that receives the gathered data (0 to
    `nranks - 1`). If `None`, performs an all-gather.
    Defaults to `None`.
  * **stream** – CUDA stream for the operation. Defaults to `None` (the
    default stream).
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If send and receive buffers have mismatched dtypes,
      `recvbuf` is too small, are on the wrong device, are
      invalid specifications, or the communicator is not
      initialized.

#### SEE ALSO
[`ncclAllGather()`](../../api/colls.md#c.ncclAllGather), [`ncclGather()`](../../api/colls.md#c.ncclGather)

## scatter

#### Communicator.scatter(sendbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI | Any, recvbuf: [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, root: int, \*, stream: [Stream](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Stream.html#cuda.core.Stream) | cuda.core.typing.IsStreamType | int | None = None) → None

Scatters data from the root rank to all ranks.

Each rank receives `count` elements from the root rank. On the
root rank, `count` elements from `sendbuf + i * count` are sent
to rank `i`. `sendbuf` is not used on non-root ranks.

On the root rank, both buffers must have matching data types.
Element count is inferred from `recvbuf`: `count = recvcount`.
The root rank requires `sendcount >= nranks` and
`sendcount / nranks == recvcount`. In-place operation occurs when
`recvbuf` resolves to `sendbuf_address + root * count`.

* **Parameters:**
  * **sendbuf** – Source buffer (only used on the root rank, size
    `>= nranks * count` elements).
  * **recvbuf** – Destination buffer with `count` elements.
  * **root** – Root rank that scatters the data (0 to `nranks - 1`).
  * **stream** – CUDA stream for the operation. Defaults to `None` (the
    default stream).
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If send and receive buffers have mismatched dtypes,
      `sendbuf` is too small on the root rank, are on the wrong
      device, are invalid specifications, or the communicator is
      not initialized.

#### SEE ALSO
[`ncclScatter()`](../../api/colls.md#c.ncclScatter)

## create_pre_mul_sum

#### Communicator.create_pre_mul_sum(scalar: int | float | numpy.ndarray | [Buffer](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Buffer.html#cuda.core.Buffer) | SupportsDLPack | SupportsCAI, datatype: [NcclDataType](../types.md#nccl.core.NcclDataType) | None = None) → [CustomRedOp](../resources.md#nccl.core.CustomRedOp)

Creates a PreMulSum custom reduction operator.

Performs `output = scalar * sum(inputs)` and is useful for
averaging (`scalar = 1/N`) or weighted reductions. The returned
[`CustomRedOp`](../resources.md#nccl.core.CustomRedOp) is tracked by the communicator
and may be released explicitly via its
`close()` method, or automatically
when the communicator is destroyed or aborted.

* **Parameters:**
  * **scalar** – Scalar multiplier value. A Python int or float is
    converted to a NumPy array using host memory. A NumPy array
    must contain exactly 1 element and uses host memory. An
    `NcclSupportedBuffer` is treated as a device buffer with
    exactly 1 element.
  * **datatype** – NCCL data type of the scalar and reduction. If
    `None`, it is inferred from `scalar`: Python `int`
    becomes `int64` and Python `float` becomes `float64`
    (NumPy’s natural dtypes); a NumPy array uses the array’s
    dtype; a device buffer uses the buffer’s dtype.
* **Returns:**
  [`CustomRedOp`](../resources.md#nccl.core.CustomRedOp) for the PreMulSum operator.
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized; the scalar
      type is unsupported; the NumPy array or device buffer does
      not contain exactly 1 element; or the requested datatype
      does not match a device buffer’s dtype.

#### SEE ALSO
[`ncclRedOpCreatePreMulSum()`](../../api/ops.md#c.ncclRedOpCreatePreMulSum)
