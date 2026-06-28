# Queue Pair (QP) Specific APIs

NVSHMEM provides a set of queue pair (QP) specific communication APIs that enable fine-grained control over communication resources for advanced use cases. Queue pairs are communication endpoints that can be created and managed independently, allowing applications to separate different communication streams and potentially achieve better performance through reduced contention and improved load balancing.

## Overview

Queue pairs in NVSHMEM provide an abstraction for managing separate communication channels. Each queue pair represents an independent communication resource that can be used for RMA operations, signaling operations, and memory ordering operations. By using multiple queue pairs, applications can:

  * **Reduce contention** : Different communication patterns can use separate queue pairs to avoid contention on shared resources.
  * **Improve load balancing** : Multiple queue pairs can be assigned to different threads or workloads to balance communication resources.
  * **Enable independent ordering** : Operations on different queue pairs can proceed independently, allowing for more flexible ordering semantics.
  * **Support specialized communication patterns** : Applications with distinct communication phases can use different queue pairs for each phase.

## QP Handle

Queue pairs are identified by a handle of type `nvshmemx_qp_handle_t`. This handle must be obtained through the `nvshmemx_qp_create` function and is used as a parameter to all QP-specific communication operations. The handle is an opaque type that should not be manipulated directly by user code.

## Special QP Handle Values

In addition to handles created by `nvshmemx_qp_create`, the following special values from the `nvshmemx_qp_handle_index_t` enumeration can be used with communication operations (RMA and signaling):

  * `NVSHMEMX_QP_DEFAULT`: Use the default queue pairs. When this value is passed to a QP-specific communication API, it has the same effect as calling the corresponding non-QP-specific API. This allows applications to switch between custom and default queue pairs without code changes.
  * `NVSHMEMX_QP_ANY`: Use any available queue pair, either a default queue pair or one of the custom created queue pairs. The implementation will select an appropriate queue pair for the operation. This can be useful for load balancing when the application does not require operations to use specific queue pairs.

These special values provide flexibility in queue pair selection, allowing applications to write code that can work with or without custom queue pairs, or to allow the implementation to optimize queue pair selection.

## Special PE Values for Synchronization

For synchronization operations (fence and quiet), in addition to specifying a particular PE number, the following special value from the `nvshmem_pe_index_t` enumeration can be used:

  * `NVSHMEMX_PE_ALL` (alias `NVSHMEMX_PE_ANY`): Synchronize operations to all PEs, not just a single target PE. When this value is used with fence operations, ordering is ensured for operations to all PEs. When used with quiet operations, completion is ensured for operations to all PEs.

## Special QP Values for Synchronization

For synchronization operations (fence and quiet), in addition to the special values available for communication operations, the following additional behavior is available:

  * `NVSHMEMX_QP_DEFAULT`: For synchronization operations, this synchronizes all default queue pairs, ensuring ordering or completion for all operations on default communication resources.
  * `NVSHMEMX_QP_ALL`: For synchronization operations, this synchronizes all queue pairs (both default and custom created), ensuring ordering or completion for operations across all communication resources. Note that `NVSHMEMX_QP_ALL` is an alias for `NVSHMEMX_QP_ANY`.

These special values enable powerful synchronization patterns. For example, using `NVSHMEMX_QP_ALL` with `NVSHMEMX_PE_ALL` in a quiet operation ensures that all operations on all queue pairs to all PEs are complete and visible, providing a global completion guarantee across all communication resources.

## API Categories

The QP-specific APIs are organized into the following categories:

  * **QP Creation** : APIs for creating and managing queue pairs (NVSHMEMX_QP_CREATE).
  * **QP RMA Operations** : Remote memory access operations that target specific queue pairs, including put/get operations in various forms (scalar, vector, strided, non-blocking) and thread group variants (QP Remote Memory Access).
  * **QP Signal Operations** : Signaling operations that use specific queue pairs for point-to-point synchronization, including put-with-signal variants (QP Signaling Operations).
  * **QP Memory Ordering** : Memory ordering operations (fence and quiet) that operate on specific queue pairs to ensure ordering and completion of operations (QP Memory Ordering).

## Device-Side Only

With the exception of `nvshmemx_qp_create`, which is a host-side initialization function, all QP-specific communication APIs are device-side operations intended to be called from CUDA kernels. These APIs provide the same functionality as their non-QP counterparts but allow targeting specific queue pair resources.

## Ordering Semantics

Operations issued on different queue pairs are independent and do not have any inherent ordering relationship. To ensure ordering between operations on different queue pairs, explicit synchronization (such as `nvshmemx_qp_quiet`) must be used. Operations on the same queue pair follow the same ordering semantics as their non-QP counterparts, as described in Section Memory Ordering.

## Thread Safety

Queue pair handles can be safely shared and used by multiple threads within a CUDA kernel. However, the thread safety guarantees for QP-specific operations are the same as for non-QP operations. Thread group variants (e.g., `_warp` and `_block` suffixes) must be called collectively by all threads in the corresponding thread group.

## Transport Support and Compatibility

The QP-specific APIs are designed to be portable across different system configurations and transport implementations. Applications can safely call QP-specific APIs in all environments, including configurations where custom queue pairs are not supported or not needed.

In systems where application PEs reside within a single NVLink domain and communication occurs exclusively over NVLink, custom queue pairs are not necessary for performance. Similarly, not all remote transport plugin implementations support the creation of additional queue pairs beyond the default resources. Currently, custom queue pair creation is supported by the IBRC, IBDEVX, IBGDA (with RC queue pairs), and GPUNetIO transport plugins.

When `nvshmemx_qp_create` is called in an environment where custom queue pairs are not supported—either because the transport does not support them or because all PEs are within a single NVLink domain—the function will succeed but return `NVSHMEMX_QP_DEFAULT` as the handle value for all newly created queue pairs. This graceful fallback behavior ensures that applications remain portable without requiring conditional compilation or runtime checks. Operations performed using these handles will execute correctly using the default communication resources, maintaining functional correctness while allowing applications to be written with QP-specific APIs throughout.

This design allows developers to write code that takes advantage of custom queue pairs when available for improved performance and load balancing, while automatically falling back to default behavior on systems where custom queue pairs are not supported or not beneficial.

### **NVSHMEMX_QP_CREATE**

`int nvshmemx_qp_create ( int num_qps , nvshmemx_qp_handle_t **out_qp_array )`

_num_qps [IN]_
    The number of queue pairs to create.
_out_qp_array [OUT]_
    Pointer to an array of queue pair handles that will be allocated and populated by this function.

**Description**

The `nvshmemx_qp_create` function creates `num_qps` queue pair handles and returns them in a newly allocated array. The function allocates memory for the array of handles and stores the pointer in `out_qp_array`.

**Returns**

Returns 0 on success. Returns a non-zero value on failure.

**Notes**

**Collective Operation:** This is a collective operation that must be called by all PEs in the world team. All PEs must call this function with the same `num_qps` value.

**PE-Independent Handles:** Each queue pair handle returned in the array can be used to communicate with any PE. The handles are not specific to a particular target PE. For example, a single queue pair handle can be used for RMA operations targeting PE 0, PE 1, PE 2, etc.

**Per-PE Synchronization:** Although a single queue pair handle can target multiple PEs, synchronization operations (fence and quiet) on that queue pair can be performed on a per-PE basis. This allows fine-grained control over ordering and completion guarantees for operations to specific PEs on specific queue pairs.

**Memory Management:** The library allocates the array pointed to by `out_qp_array`. Users must not allocate memory for this array before calling the function. Users are responsible for freeing this memory after NVSHMEM finalization. Do not free the array before finalization as the handles must remain valid.

This function must be called on the host after NVSHMEM initialization and before any device-side QP-specific operations are performed.

The number of queue pairs that can be created may be limited by system resources. Applications should handle creation failures gracefully.

Queue pairs created by this function are independent of the default communication context and do not interfere with operations performed without QP handles. Operations using custom queue pair handles and operations using the default context (non-QP APIs) operate on separate resources.

## QP Remote Memory Access

The following routines provide RMA operations on specific queue pairs. These operations have the same semantics as their non-QP counterparts but target a specific queue pair resource for improved control over communication channels.

### **NVSHMEMX_QP_PUT**

`__device__ void nvshmemx_qp_TYPENAME_put ( TYPE *dest , const TYPE *source , size_t nelems , int pe , nvshmemx_qp_handle_t qp_index )`

`__device__ void nvshmemx_qp_TYPENAME_put_block ( TYPE *dest , const TYPE *source , size_t nelems , int pe , nvshmemx_qp_handle_t qp_index )`

`__device__ void nvshmemx_qp_TYPENAME_put_warp ( TYPE *dest , const TYPE *source , size_t nelems , int pe , nvshmemx_qp_handle_t qp_index )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

_dest [OUT]_
    Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.
_source [IN]_
    Device address of the data object containing the data to be copied. The type of `source` should match that implied in the SYNOPSIS section.
_nelems [IN]_
    Number of elements in the `dest` and `source` arrays.
_pe [IN]_
    PE number of the remote PE.
_qp_index [IN]_
    Queue pair handle identifying the communication resource to use. This can be a handle obtained from `nvshmemx_qp_create`, or one of the special values `NVSHMEMX_QP_DEFAULT` or `NVSHMEMX_QP_ANY`.

**Description**

The QP-specific put routines perform the same operation as their non-QP counterparts but use a specific queue pair resource. The routines return after the data has been copied out of the `source` array on the local PE. The delivery of data words into the data object on the destination PE may occur in any order. Furthermore, two successive put routines on the same queue pair may deliver data out of order unless a call to `nvshmemx_qp_fence` is introduced between the two calls.

Operations on different queue pairs are independent and do not have any inherent ordering relationship.

**Thread Group Variants:**

Thread group variants with `_warp` and `_block` suffixes are also available and must be called collectively by all active threads in the warp or block, respectively:

  * `nvshmemx_qp_TYPENAME_put_warp`
  * `nvshmemx_qp_TYPENAME_put_block`

**Returns**

None.

### **NVSHMEMX_QP_P**

`__device__ void nvshmemx_qp_TYPENAME_p ( TYPE *dest , TYPE value , int pe , nvshmemx_qp_handle_t qp_index )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

_dest [OUT]_
    Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.
_value [IN]_
    The value to be transferred to `dest`. The type of `value` should match that implied in the SYNOPSIS section.
_pe [IN]_
    The number of the remote PE.
_qp_index [IN]_
    Queue pair handle identifying the communication resource to use. This can be a handle obtained from `nvshmemx_qp_create`, or one of the special values `NVSHMEMX_QP_DEFAULT` or `NVSHMEMX_QP_ANY`.

**Description**

The QP-specific scalar put routines provide a very low latency put capability for single elements of most basic types using a specific queue pair resource. These routines perform the same operation as their non-QP counterparts but use the specified queue pair.

As with `nvshmemx_qp_put`, these routines start the remote transfer and may return before the data is delivered to the remote PE. Use `nvshmemx_qp_quiet` to force completion of all remote _Put_ transfers on the specified queue pairs.

Operations on different queue pairs are independent and do not have any inherent ordering relationship.

**Returns**

None.

### **NVSHMEMX_QP_GET**

`__device__ void nvshmemx_qp_TYPENAME_get ( TYPE *dest , const TYPE *source , size_t nelems , int pe , nvshmemx_qp_handle_t qp_index )`

`__device__ void nvshmemx_qp_TYPENAME_get_block ( TYPE *dest , const TYPE *source , size_t nelems , int pe , nvshmemx_qp_handle_t qp_index )`

`__device__ void nvshmemx_qp_TYPENAME_get_warp ( TYPE *dest , const TYPE *source , size_t nelems , int pe , nvshmemx_qp_handle_t qp_index )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

_dest [OUT]_
    Device address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.
_source [IN]_
    Symmetric address of the source data object. The type of `source` should match that implied in the SYNOPSIS section.
_nelems [IN]_
    Number of elements in the `dest` and `source` arrays.
_pe [IN]_
    PE number of the remote PE.
_qp_index [IN]_
    Queue pair handle identifying the communication resource to use. This can be a handle obtained from `nvshmemx_qp_create`, or one of the special values `NVSHMEMX_QP_DEFAULT` or `NVSHMEMX_QP_ANY`.

**Description**

The QP-specific get routines perform the same operation as their non-QP counterparts but use a specific queue pair resource. The routines return after the data has been copied into the `dest` array on the local PE.

Operations on different queue pairs are independent and do not have any inherent ordering relationship.

**Thread Group Variants:**

Thread group variants with `_warp` and `_block` suffixes are also available and must be called collectively by all active threads in the warp or block, respectively:

  * `nvshmemx_qp_TYPENAME_get_warp`
  * `nvshmemx_qp_TYPENAME_get_block`

**Returns**

None.

### **NVSHMEMX_QP_G**

`__device__ TYPE nvshmemx_qp_TYPENAME_g ( const TYPE *source , int pe , nvshmemx_qp_handle_t qp_index )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

_source [IN]_
    Symmetric address of the source data object. The type of `source` should match that implied in the SYNOPSIS section.
_pe [IN]_
    The number of the remote PE.
_qp_index [IN]_
    Queue pair handle identifying the communication resource to use. This can be a handle obtained from `nvshmemx_qp_create`, or one of the special values `NVSHMEMX_QP_DEFAULT` or `NVSHMEMX_QP_ANY`.

**Description**

The QP-specific scalar get routines provide a very low latency get capability for single elements of most basic types using a specific queue pair resource. These routines perform the same operation as their non-QP counterparts but use the specified queue pair.

The routine returns the value retrieved from the remote PE.

Operations on different queue pairs are independent and do not have any inherent ordering relationship.

**Returns**

Returns the value retrieved from the remote PE. The return type matches the _TYPE_ specified in the function name.

### **NVSHMEMX_QP_PUT_NBI**

`__device__ void nvshmemx_qp_TYPENAME_put_nbi ( TYPE *dest , const TYPE *source , size_t nelems , int pe , nvshmemx_qp_handle_t qp_index )`

`__device__ void nvshmemx_qp_TYPENAME_put_nbi_block ( TYPE *dest , const TYPE *source , size_t nelems , int pe , nvshmemx_qp_handle_t qp_index )`

`__device__ void nvshmemx_qp_TYPENAME_put_nbi_warp ( TYPE *dest , const TYPE *source , size_t nelems , int pe , nvshmemx_qp_handle_t qp_index )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

_dest [OUT]_
    Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.
_source [IN]_
    Device address of the data object containing the data to be copied. The type of `source` should match that implied in the SYNOPSIS section.
_nelems [IN]_
    Number of elements in the `dest` and `source` arrays.
_pe [IN]_
    PE number of the remote PE.
_qp_index [IN]_
    Queue pair handle identifying the communication resource to use. This can be a handle obtained from `nvshmemx_qp_create`, or one of the special values `NVSHMEMX_QP_DEFAULT` or `NVSHMEMX_QP_ANY`.

**Description**

The QP-specific nonblocking put routines perform the same operation as their non-QP counterparts but use a specific queue pair resource. The routines initiate the transfer and return immediately without waiting for the operation to complete.

The completion of the transfer can be ensured by calling `nvshmemx_qp_quiet` on the appropriate queue pairs.

Operations on different queue pairs are independent and do not have any inherent ordering relationship.

**Thread Group Variants:**

Thread group variants with `_warp` and `_block` suffixes are also available and must be called collectively by all active threads in the warp or block, respectively:

  * `nvshmemx_qp_TYPENAME_put_nbi_warp`
  * `nvshmemx_qp_TYPENAME_put_nbi_block`

**Returns**

None.

### **NVSHMEMX_QP_GET_NBI**

`__device__ void nvshmemx_qp_TYPENAME_get_nbi ( TYPE *dest , const TYPE *source , size_t nelems , int pe , nvshmemx_qp_handle_t qp_index )`

`__device__ void nvshmemx_qp_TYPENAME_get_nbi_block ( TYPE *dest , const TYPE *source , size_t nelems , int pe , nvshmemx_qp_handle_t qp_index )`

`__device__ void nvshmemx_qp_TYPENAME_get_nbi_warp ( TYPE *dest , const TYPE *source , size_t nelems , int pe , nvshmemx_qp_handle_t qp_index )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

_dest [OUT]_
    Device address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.
_source [IN]_
    Symmetric address of the source data object. The type of `source` should match that implied in the SYNOPSIS section.
_nelems [IN]_
    Number of elements in the `dest` and `source` arrays.
_pe [IN]_
    PE number of the remote PE.
_qp_index [IN]_
    Queue pair handle identifying the communication resource to use. This can be a handle obtained from `nvshmemx_qp_create`, or one of the special values `NVSHMEMX_QP_DEFAULT` or `NVSHMEMX_QP_ANY`.

**Description**

The QP-specific nonblocking get routines perform the same operation as their non-QP counterparts but use a specific queue pair resource. The routines initiate the transfer and return immediately without waiting for the operation to complete.

The completion of the transfer can be ensured by calling `nvshmemx_qp_quiet` on the appropriate queue pairs.

Operations on different queue pairs are independent and do not have any inherent ordering relationship.

**Thread Group Variants:**

Thread group variants with `_warp` and `_block` suffixes are also available and must be called collectively by all active threads in the warp or block, respectively:

  * `nvshmemx_qp_TYPENAME_get_nbi_warp`
  * `nvshmemx_qp_TYPENAME_get_nbi_block`

**Returns**

None.

## QP Signaling Operations

The following routines provide signaling operations on specific queue pairs. These operations allow for point-to-point synchronization using specific queue pair resources.

### **NVSHMEMX_QP_SIGNAL_OP**

`__device__ void nvshmemx_qp_signal_op ( uint64_t *sig_addr , uint64_t signal , int sig_op , int pe , nvshmemx_qp_handle_t qp_index )`

_sig_addr [OUT]_
    Symmetric address of the signal data object on the remote PE.
_signal [IN]_
    The value to be used for the signal update.
_sig_op [IN]_
    The signal operator to be applied. Valid operators are `NVSHMEM_SIGNAL_SET` and `NVSHMEM_SIGNAL_ADD`.
_pe [IN]_
    PE number of the remote PE.
_qp_index [IN]_
    Queue pair handle identifying the communication resource to use. This can be a handle obtained from `nvshmemx_qp_create`, or one of the special values `NVSHMEMX_QP_DEFAULT` or `NVSHMEMX_QP_ANY`.

**Description**

The `nvshmemx_qp_signal_op` routine performs a signal operation on the signal data object at `sig_addr` on PE `pe` using the specified queue pair. The operation performed depends on the `sig_op` parameter:

  * `NVSHMEM_SIGNAL_SET`: The signal data object is set to the value of `signal`.
  * `NVSHMEM_SIGNAL_ADD`: The value of `signal` is added to the signal data object.

The signal operation is performed atomically with respect to other signal operations, signal-fetch operations, and point-to-point synchronization routines that access the same signal data object.

Operations on different queue pairs are independent and do not have any inherent ordering relationship.

**Returns**

None.

### **NVSHMEMX_QP_PUT_SIGNAL**

`__device__ void nvshmemx_qp_TYPENAME_put_signal ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe , nvshmemx_qp_handle_t qp_index )`

`__device__ void nvshmemx_qp_TYPENAME_put_signal_block ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe , nvshmemx_qp_handle_t qp_index )`

`__device__ void nvshmemx_qp_TYPENAME_put_signal_warp ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe , nvshmemx_qp_handle_t qp_index )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

_dest [OUT]_
    Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.
_source [IN]_
    Device address of the data object containing the data to be copied. The type of `source` should match that implied in the SYNOPSIS section.
_nelems [IN]_
    Number of elements in the `dest` and `source` arrays.
_sig_addr [OUT]_
    Symmetric address of the signal data object on the remote PE.
_signal [IN]_
    The value to be used for the signal update.
_sig_op [IN]_
    The signal operator to be applied. Valid operators are `NVSHMEM_SIGNAL_SET` and `NVSHMEM_SIGNAL_ADD`.
_pe [IN]_
    PE number of the remote PE.
_qp_index [IN]_
    Queue pair handle identifying the communication resource to use. This can be a handle obtained from `nvshmemx_qp_create`, or one of the special values `NVSHMEMX_QP_DEFAULT` or `NVSHMEMX_QP_ANY`.

**Description**

The QP-specific put-with-signal routines perform the same operation as their non-QP counterparts but use a specific queue pair resource. The routines copy the data from `source` to `dest` on PE `pe`, and after the data transfer completes, perform a signal operation on the signal data object at `sig_addr`.

The signal operation is guaranteed to occur after the data transfer completes. The signal operation is performed atomically with respect to other signal operations, signal-fetch operations, and point-to-point synchronization routines that access the same signal data object.

Operations on different queue pairs are independent and do not have any inherent ordering relationship.

**Thread Group Variants:**

Thread group variants with `_warp` and `_block` suffixes are also available and must be called collectively by all active threads in the warp or block, respectively:

  * `nvshmemx_qp_TYPENAME_put_signal_warp`
  * `nvshmemx_qp_TYPENAME_put_signal_block`

**Returns**

None.

### **NVSHMEMX_QP_PUT_SIGNAL_NBI**

`__device__ void nvshmemx_qp_TYPENAME_put_signal_nbi ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe , nvshmemx_qp_handle_t qp_index )`

`__device__ void nvshmemx_qp_TYPENAME_put_signal_nbi_block ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe , nvshmemx_qp_handle_t qp_index )`

`__device__ void nvshmemx_qp_TYPENAME_put_signal_nbi_warp ( TYPE *dest , const TYPE *source , size_t nelems , uint64_t *sig_addr , uint64_t signal , int sig_op , int pe , nvshmemx_qp_handle_t qp_index )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

_dest [OUT]_
    Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.
_source [IN]_
    Device address of the data object containing the data to be copied. The type of `source` should match that implied in the SYNOPSIS section.
_nelems [IN]_
    Number of elements in the `dest` and `source` arrays.
_sig_addr [OUT]_
    Symmetric address of the signal data object on the remote PE.
_signal [IN]_
    The value to be used for the signal update.
_sig_op [IN]_
    The signal operator to be applied. Valid operators are `NVSHMEM_SIGNAL_SET` and `NVSHMEM_SIGNAL_ADD`.
_pe [IN]_
    PE number of the remote PE.
_qp_index [IN]_
    Queue pair handle identifying the communication resource to use. This can be a handle obtained from `nvshmemx_qp_create`, or one of the special values `NVSHMEMX_QP_DEFAULT` or `NVSHMEMX_QP_ANY`.

**Description**

The QP-specific nonblocking put-with-signal routines perform the same operation as their non-QP counterparts but use a specific queue pair resource. The routines initiate the data transfer from `source` to `dest` on PE `pe`, followed by a signal operation on the signal data object at `sig_addr`. The routine returns immediately without waiting for the operations to complete.

The signal operation is guaranteed to occur after the data transfer completes. The signal operation is performed atomically with respect to other signal operations, signal-fetch operations, and point-to-point synchronization routines that access the same signal data object.

The completion of both the data transfer and signal operation can be ensured by calling `nvshmemx_qp_quiet` on the appropriate queue pairs.

Operations on different queue pairs are independent and do not have any inherent ordering relationship.

**Thread Group Variants:**

Thread group variants with `_warp` and `_block` suffixes are also available and must be called collectively by all active threads in the warp or block, respectively:

  * `nvshmemx_qp_TYPENAME_put_signal_nbi_warp`
  * `nvshmemx_qp_TYPENAME_put_signal_nbi_block`

**Returns**

None.

## QP Memory Ordering

The following routines provide memory ordering operations on specific queue pairs. These operations ensure ordering and completion of operations issued on the specified queue pairs.

### **NVSHMEMX_QP_FENCE**

`__device__ void nvshmemx_qp_fence ( int pe , nvshmemx_qp_handle_t *qp_handle , int num_qps )`

`__device__ void nvshmemx_qp_fence_block ( int pe , nvshmemx_qp_handle_t *qp_handle , int num_qps )`

`__device__ void nvshmemx_qp_fence_warp ( int pe , nvshmemx_qp_handle_t *qp_handle , int num_qps )`

_pe [IN]_
    PE number of the target PE for which ordering is to be ensured. This can be a specific PE number, or the special value `NVSHMEMX_PE_ALL` (alias `NVSHMEMX_PE_ANY`) to ensure ordering for all PEs.
_qp_handle [IN]_
    Pointer to an array of queue pair handles. This can also be a pointer to one of the special values `NVSHMEMX_QP_DEFAULT` or `NVSHMEMX_QP_ALL` to fence all default queue pairs or all queue pairs (including custom created ones), respectively.
_num_qps [IN]_
    Number of queue pairs in the `qp_handle` array. When using special QP values (`NVSHMEMX_QP_DEFAULT` or `NVSHMEMX_QP_ALL`), this should be set to 1.

**Description**

The `nvshmemx_qp_fence` routine ensures ordering of delivery of operations on symmetric data objects issued on the specified queue pairs to the target PE(s). All operations on symmetric data objects issued to PE `pe` on the specified queue pairs prior to the call to `nvshmemx_qp_fence` are guaranteed to be delivered before any subsequent operations on symmetric data objects to the same PE(s) on the same queue pairs.

`nvshmemx_qp_fence` guarantees order of delivery, not completion. It does not guarantee order of delivery of nonblocking _Get_ or values fetched by nonblocking AMO routines.

The routine operates on all queue pairs specified in the `qp_handle` array. Ordering is ensured for operations on these queue pairs to the specified PE(s).

**Special PE Values:**

When `pe` is set to `NVSHMEMX_PE_ALL` (or its alias `NVSHMEMX_PE_ANY`), ordering is ensured for operations to all PEs, not just a single target PE.

**Special QP Values:**

  * `NVSHMEMX_QP_DEFAULT`: Fence all default queue pairs. Pass a pointer to this value as `qp_handle` and set `num_qps` to 1.
  * `NVSHMEMX_QP_ALL`: Fence all queue pairs (both default and custom created). Pass a pointer to this value as `qp_handle` and set `num_qps` to 1. This ensures ordering across all communication resources.

**Thread Group Variants:**

Thread group variants with `_warp` and `_block` suffixes are also available and must be called collectively by all active threads in the warp or block, respectively:

  * `nvshmemx_qp_fence_warp`: Collective fence operation for a warp.
  * `nvshmemx_qp_fence_block`: Collective fence operation for a thread block.

**Returns**

None.

**Notes**

Thread group variants provide potential performance benefits through cooperative synchronization.

### **NVSHMEMX_QP_QUIET**

`__device__ void nvshmemx_qp_quiet ( int pe , nvshmemx_qp_handle_t *qp_handle , int num_qps )`

`__device__ void nvshmemx_qp_quiet_block ( int pe , nvshmemx_qp_handle_t *qp_handle , int num_qps )`

`__device__ void nvshmemx_qp_quiet_warp ( int pe , nvshmemx_qp_handle_t *qp_handle , int num_qps )`

_pe [IN]_
    PE number of the target PE for which completion is to be ensured. This can be a specific PE number, or the special value `NVSHMEMX_PE_ALL` (alias `NVSHMEMX_PE_ANY`) to ensure completion for operations to all PEs.
_qp_handle [IN]_
    Pointer to an array of queue pair handles. This can also be a pointer to one of the special values `NVSHMEMX_QP_DEFAULT` or `NVSHMEMX_QP_ALL` to quiet all default queue pairs or all queue pairs (including custom created ones), respectively.
_num_qps [IN]_
    Number of queue pairs in the `qp_handle` array. When using special QP values (`NVSHMEMX_QP_DEFAULT` or `NVSHMEMX_QP_ALL`), this should be set to 1.

**Description**

The `nvshmemx_qp_quiet` routine ensures completion of all operations on symmetric data objects issued to PE(s) `pe` on the specified queue pairs. All operations on symmetric data objects targeting the specified PE(s) on the given queue pairs are guaranteed to be complete and visible when `nvshmemx_qp_quiet` returns.

The routine operates on all queue pairs specified in the `qp_handle` array. Completion is ensured for operations on these queue pairs to the specified PE(s).

On systems with only NVLink, all operations on symmetric data objects are guaranteed to be complete and visible to the target PE(s) when `nvshmemx_qp_quiet` returns. On systems with both NVLink and InfiniBand, visibility is only guaranteed at the destination PE(s).

**Special PE Values:**

When `pe` is set to `NVSHMEMX_PE_ALL` (or its alias `NVSHMEMX_PE_ANY`), completion is ensured for operations to all PEs, not just a single target PE.

**Special QP Values:**

  * `NVSHMEMX_QP_DEFAULT`: Quiet all default queue pairs. Pass a pointer to this value as `qp_handle` and set `num_qps` to 1.
  * `NVSHMEMX_QP_ALL`: Quiet all queue pairs (both default and custom created). Pass a pointer to this value as `qp_handle` and set `num_qps` to 1. This ensures completion of operations across all communication resources.

**Thread Group Variants:**

Thread group variants with `_warp` and `_block` suffixes are also available and must be called collectively by all active threads in the warp or block, respectively:

  * `nvshmemx_qp_quiet_warp`: Collective quiet operation for a warp.
  * `nvshmemx_qp_quiet_block`: Collective quiet operation for a thread block.

**Returns**

None.

**Notes**

Thread group variants provide potential performance benefits through cooperative synchronization but require collective invocation by all threads in the group.
