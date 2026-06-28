# Signaling Operations

NVSHMEM provides signaling operations that can be used to update a remote flag variable. When used in conjunction with wait/test routines at the remote PE, these routines can provide efficient point-to-point synchronization. The following example shows signal operations used to implement neighbor communication in a ring.

    nvshmem_putmem(dest, src, size, (pe+1) % npes);
    nvshmem_quiet();
    nvshmemx_int_signal(flag, 1, (pe+1) % npes);
    nvshmem_int_wait_until(flag, NVSHMEM_CMP_EQ, 1);

his section specifies the NVSHMEM support for _put-with-signal_ , nonblocking _put-with-signal_ , and _signal-fetch_ routines. The put-with-signal routines provide a method for copying data from a contiguous local data object to a data object on a specified PE and subsequently updating a remote flag to signal completion. The signal-fetch routine provides support for fetching a signal update operation.

## Atomicity Guarantees for Signaling Operations

All signaling operations put-with-signal, nonblocking put-with-signal, and signal-fetch are performed on a signal data object, a remotely accessible symmetric object of type `uint64_t`. A signal operator in the put-with-signal routine is a NVSHMEM library constant that determines the type of update to be performed as a signal on the signal data object.

All signaling operations complete as if performed atomically with respect to the following:

  * other signal operations that update the signal data object using the same datatype;
  * signal-fetch routine that fetches the signal data object; and
  * any point-to-point synchronization routine that accesses the signal data object using the same datatype.

## Available Signal Operators

With the atomicity guarantees as described in Section Atomicity Guarantees for Signaling Operations, the following options can be used as a signal operator.

`NVSHMEM_SIGNAL_SET`
    An update to signal data object is an atomic set operation. It writes an unsigned 64-bit value as a signal into the signal data object on a remote `PE` as an atomic operation.
`NVSHMEM_SIGNAL_ADD`
    An update to signal data object is an atomic add operation. It adds an unsigned 64-bit value as a signal into the signal data object on a remote `PE` as an atomic operation.

## **NVSHMEM_PUT_SIGNAL**

`void nvshmem_TYPENAME_put_signal ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`void nvshmemx_TYPENAME_put_signal_on_stream ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe , cudaStream_t stream )`

`__device__ void nvshmem_TYPENAME_put_signal ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`__device__ void nvshmemx_TYPENAME_put_signal_block ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`__device__ void nvshmemx_TYPENAME_put_signal_warp ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

`void nvshmem_putSIZE_signal ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`void nvshmemx_putSIZE_signal_on_stream ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe , cudaStream_t stream )`

`__device__ void nvshmem_putSIZE_signal ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`__device__ void nvshmemx_putSIZE_signal_block ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`__device__ void nvshmemx_putSIZE_signal_warp ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

where _SIZE_ is one of `8, 16, 32, 64, 128`.

`void nvshmem_putmem_signal ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`void nvshmemx_putmem_signal_on_stream ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe , cudaStream_t stream )`

`__device__ void nvshmem_putmem_signal ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`__device__ void nvshmemx_putmem_signal_block ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`__device__ void nvshmemx_putmem_signal_warp ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

Symmetric address of the data object to be updated on the remote PE. The type of `dest` should match that implied in the SYNOPSIS section.

Symmetric address or host/device address registered via `nvshmemx_buffer_register` of data object containing the data to be copied. The type of `source` should match that implied in the SYNOPSIS section.Additionally, it can also be backed by device shared memory when devices are connected via peer-to-peer transport.

Number of elements in the `dest` and `source` arrays. For `nvshmem_putmem_signal` and `nvshmem_ctx_putmem_signal`, elements are bytes.

Symmetric address of the signal data object to be updated on the remote PE as a signal.

Unsigned 64-bit value that is used for updating the remote `sig_addr` signal data object.

Signal operator that represents the type of update to be performed on the remote `sig_addr` signal data object.

PE number of the remote PE.

**Description**

The _put-with-signal_ routines provide a method for copying data from a contiguous local data object to a data object on a specified PE and subsequently updating a remote flag to signal completion. The routines return after the data has been copied out of the `source` array on the local PE.

The `sig_op` signal operator determines the type of update to be performed on the remote `sig_addr` signal data object. The completion of signal update based on the `sig_op` signal operator using the `signal` flag on the remote PE indicates the delivery of its corresponding `dest` data words into the data object on the remote PE.

An update to the `sig_addr` signal data object through a _put-with-signal_ routine completes as if performed atomically as described in Section Atomicity Guarantees for Signaling Operations. The various options as described in Section Available Signal Operators can be used as the `sig_op` signal operator.

**Returns**

None.

**Notes**

The `dest` and `sig_addr` data objects must both be remotely accessible.

`sig_addr` and `dest` may not be overlapping in memory.

The completion of signal update using the `signal` flag on the remote PE indicates only the delivery of its corresponding `dest` data words into the data object on the remote PE. Without a memory-ordering operation, there is no implied ordering between the signal update of a _put-with-signal_ routine and another data transfer. For example, the completion of the signal update in a sequence consisting of a put routine followed by a _put-with-signal_ routine does not imply delivery of the put routine’s data.

The following example demonstrates the usage of `nvshmem_put_signal`. It shows the implementation of a broadcast operation from PE 0 to itself and all other PEs in the job as a simple ring-based algorithm using `nvshmem_put_signal`: ./example_code/shmem_put_signal_example.c

## **NVSHMEM_PUT_SIGNAL_NBI**

`void nvshmem_TYPENAME_put_signal_nbi ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`void nvshmemx_TYPENAME_put_signal_nbi_on_stream ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe , cudaStream_t stream )`

`__device__ void nvshmem_TYPENAME_put_signal_nbi ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`__device__ void nvshmemx_TYPENAME_put_signal_nbi_block ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`__device__ void nvshmemx_TYPENAME_put_signal_nbi_warp ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

`void nvshmem_putSIZE_signal_nbi ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`void nvshmemx_putSIZE_signal_nbi_on_stream ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe , cudaStream_t stream )`

`__device__ void nvshmem_putSIZE_signal_nbi ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`__device__ void nvshmemx_putSIZE_signal_nbi_block ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`__device__ void nvshmemx_putSIZE_signal_nbi_warp ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

where _SIZE_ is one of `8, 16, 32, 64, 128`.

`void nvshmem_putmem_signal_nbi ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`void nvshmemx_putmem_signal_nbi_on_stream ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe , cudaStream_t stream )`

`__device__ void nvshmem_putmem_signal_nbi ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`__device__ void nvshmemx_putmem_signal_nbi_block ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

`__device__ void nvshmemx_putmem_signal_nbi_warp ( void *dest , const void *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

Symmetric address of the data object to be updated on the remote PE. The type of `dest` should match that implied in the SYNOPSIS section.

Symmetric address or host/device address registered via `nvshmemx_buffer_register` of data object containing the data to be copied. The type of `source` should match that implied in the SYNOPSIS section.Additionally, it can also be backed by device shared memory when devices are connected via peer-to-peer transport.

Number of elements in the `dest` and `source` arrays. For `nvshmem_putmem_signal_nbi` and `nvshmem_ctx_putmem_signal_nbi`, elements are bytes.

Symmetric address of the signal data object to be updated on the remote PE as a signal.

Unsigned 64-bit value that is used for updating the remote `sig_addr` signal data object.

Signal operator that represents the type of update to be performed on the remote `sig_addr` signal data object.

PE number of the remote PE.

**Description**

The nonblocking _put-with-signal_ routines provide a method for copying data from a contiguous local data object to a data object on a specified PE and subsequently updating a remote flag to signal completion.

The routines return after initiating the operation. The operation is considered complete after a subsequent call to `nvshmem_quiet`. At the completion of `nvshmem_quiet`, the data has been copied out of the `source` array on the local PE and delivered into the `dest` array on the destination PE.

The delivery of `signal` flag on the remote PE indicates only the delivery of its corresponding `dest` data words into the data object on the remote PE. Furthermore, two successive nonblocking _put-with-signal_ routines, or a nonblocking _put-with-signal_ routine with another data transfer may deliver data out of order unless a call to `nvshmem_fence` is introduced between the two calls.

The `sig_op` signal operator determines the type of update to be performed on the remote `sig_addr` signal data object.

An update to the `sig_addr` signal data object through a nonblocking _put-with-signal_ routine completes as if performed atomically as described in Section Atomicity Guarantees for Signaling Operations. The various options as described in Section Available Signal Operators can be used as the `sig_op` signal operator.

**Returns**

None.

**Notes**

The `dest` and `sig_addr` data objects must both be remotely accessible.

`sig_addr` and `dest` may not be overlapping in memory.

## **NVSHMEM_SIGNAL_FETCH**

`__device__ uint64_t nvshmem_signal_fetch ( const uint64_t *sig_addr )`

_sig_addr [IN]_
    Local address of the remotely accessible signal variable.

**Description**

`nvshmem_signal_fetch` performs a fetch operation and returns the contents of the `sig_addr` signal data object. Access to `sig_addr` signal object at the calling PE is expected to satisfy the atomicity guarantees as described in Section Atomicity Guarantees for Signaling Operations.

**Returns**

Returns the contents of the signal data object, `sig_addr`, at the calling PE.

The following datatypes are supported by the NVSHMEM signal operation.

Signal Types and Names _TYPE_ | _TYPENAME_
---|---
short | short
int | int
long | long
long long | longlong
unsigned short | ushort
unsigned int | uint
unsigned long | ulong
unsigned long long | ulonglong
int32_t | int32
int64_t | int64
uint32_t | uint32
uint64_t | uint64
size_t | size
ptrdiff_t | ptrdiff

## **NVSHMEMX_SIGNAL**

Deprecated, see `nvshmemx_signal_op` or `nvshmem_*_atomic_set`.

`__device__ inline void nvshmemx_TYPENAME_signal ( TYPE *dest , const TYPE value , int pe )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table signaltypes.

_dest [OUT]_
    Symmetric address of the signal word to be updated.
_value [IN]_
    The value to be placed in _dest_.
_pe [IN]_
    PE number of the remote PE.

**Description**

The _nvshmemx_signal_ operation atomically sets _dest_ to _value_ on the specified PE. This operation can be used together with wait and test routines for efficient point-to-point synchronization.

**Returns**

None.

## **NVSHMEMX_SIGNAL_OP**

`__device__ inline void nvshmemx_signal_op ( uint64_t *sig_addr , uint64_t signal , int sig_op , int pe )`

_sig_addr [OUT]_
    Symmetric address of the signal word to be updated.
_signal [IN]_
    The value used to update _sig_addr_.
_sig_op [IN]_
    Operation used to update _sig_addr_ with _signal_.
_pe [IN]_
    PE number of the remote PE.

**Description**

The _nvshmemx_signal_op_ operation atomically updates _sig_addr_ with _signal_ using operation _sig_op_ on the specified PE. This operation can be used together with wait and test routines for efficient point-to-point synchronization.

**Returns**

None.
