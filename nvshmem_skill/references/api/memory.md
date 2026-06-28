# Memory Management

NVSHMEM provides a set of APIs for managing the symmetric heap. The APIs allow one to dynamically allocate, deallocate, reallocate and align symmetric data objects in the symmetric heap.

## **NVSHMEM_MALLOC, NVSHMEM_FREE, NVSHMEM_ALIGN**

`void * nvshmem_malloc ( size_t size )`

`void nvshmem_free ( void *ptr )`

`void * nvshmem_align ( size_t alignment , size_t size )`

_size [IN]_
    The size, in bytes, of a block to be allocated from the symmetric heap.
_ptr [IN]_
    Symmetric address of an object in the symmetric heap.
_alignment [IN]_
    Byte alignment of the block allocated from the symmetric heap.

**Description**

The `nvshmem_malloc`, `nvshmem_free`, and `nvshmem_align` routines are collective operations that require participation by all PEs.

The `nvshmem_malloc` routine returns the symmetric address of a block of at least `size` bytes, which shall be suitably aligned so that it may be assigned to a pointer to any type of object. This space is allocated from the symmetric heap (in contrast to `malloc`, which allocates from the private heap). When `size` is zero, the `nvshmem_malloc` routine performs no action and returns a null pointer.

The `nvshmem_align` routine allocates a block in the symmetric heap that has a byte alignment specified by the `alignment` argument. The value of `alignment` shall be a multiple of `sizeof(void *)` that is also a power of two. Otherwise, the behavior is undefined. When `size` is zero, the `nvshmem_align` routine performs no action and returns a null pointer.

The `nvshmem_free` routine causes the block to which `ptr` points to be deallocated, that is, made available for further allocation. If `ptr` is a null pointer, no action is performed.

The `nvshmem_malloc`, `nvshmem_align`, and `nvshmem_free` routines are provided so that multiple PEs in a program can allocate symmetric, remotely accessible memory blocks. These memory blocks can then be used with NVSHMEM communication routines. When no action is performed, these routines return without performing a barrier. Otherwise, each of these routines includes at least one call to a procedure that is semantically equivalent to `nvshmem_barrier_all`: `nvshmem_malloc` and `nvshmem_align` call a barrier on exit; and `nvshmem_free` calls a barrier on entry. This ensures that all PEs participate in the memory allocation, and that the memory on other PEs can be used as soon as the local PE returns. The user is also responsible for calling these routines with identical argument(s) on all PEs; if differing `ptr`, `size`, or `alignment` arguments are used, the behavior of the call and any subsequent NVSHMEM calls is undefined.

**Returns**

The `nvshmem_malloc` routine returns the symmetric address of the allocated space; otherwise, it returns a null pointer.

The `nvshmem_free` routine returns no value.

The `nvshmem_align` routine returns an aligned symmetric address whose value is a multiple of `alignment`; otherwise, it returns a null pointer.

**Notes**

NVSHMEM supports both dynamic and static symmetric heap allocation policies. Dynamic symmetric heap allocation is accomplished using the CUDA VMM APIs and is enabled by default. Setting `NVSHMEM_DISABLE_CUDA_VMM` will cause NVSHMEM to use a statically allocated symmetric heap. In this mode, the total size of the symmetric heap is determined at job startup. One can specify the size of the heap using the `NVSHMEM_SYMMETRIC_SIZE` environment variable (where available).

The `nvshmem_malloc`, and `nvshmem_free` routines differ from the private heap allocation routines in that all PEs in a program must call them (a barrier is used to ensure this).

## **NVSHMEM_CALLOC**

`void * nvshmem_calloc ( size_t count , size_t size )`

_count [IN]_
    The number of elements to allocate.
_size [IN]_
    The size in bytes of each element to allocate.

**Description**

The `nvshmem_calloc` routine is a collective operation on the world team that allocates a region of remotely-accessible memory for an array of `count` objects of `size` bytes each and returns a pointer to the lowest byte address of the allocated symmetric memory. The space is initialized to all bits zero.

If the allocation succeeds, the pointer returned shall be suitably aligned so that it may be assigned to a pointer to any type of object. If the allocation does not succeed, or either `count` or `size` is `0`, the return value is a null pointer.

The values for `count` and `size` shall each be equal across all PEs calling `nvshmem_calloc`; otherwise, the behavior is undefined.

When `count` or `size` is `0`, the `nvshmem_calloc` routine returns without performing a barrier. Otherwise, this routine calls a procedure that is semantically equivalent to `nvshmem_barrier_all` on exit.

**Returns**

The `nvshmem_calloc` routine returns a pointer to the lowest byte address of the allocated space; otherwise, it returns a null pointer.

## Memory Registration

### **NVSHMEMX_BUFFER_REGISTER**

`int nvshmemx_buffer_register ( void *addr , size_t length )`

_addr [IN]_
    The address at the start of the buffer to register.
_length [IN]_
    The length of the registration.

**Description**

The _nvshmemx_buffer_register_ function registers the buffer with the remote transport (either UCX or IB) and with CUDA if the address supplied is unregistered host memory. NVSHMEM heap memory, i.e. any memory allocated with nvshmem_malloc or nvshmem_calloc, should not be registered using this function as the heap is registered by default. In fact, attempting to register nvshmem heap memory with this function will result in an error. Once memory is registered with this function, it is possible to use that memory as the local operand in any host or device AMO or RMA operations. Memory should be registered in as large of chunks as possible for maximum performance in the I/O path.

**Limitations** Using memory registered with this function as the remote operand in any nvshmem API is unsupported.

Sending a buffer that is spread over multiple registrations to a single nvshmem RMA operation is not supported.

Registering CUDA managed memory with this API is not supported.

Freeing registered memory before unregistering it is unsupported and will result in undefined behavior.

**Returns** 0 on success. NVSHMEMX_ERROR_INVALID_VALUE if the buffer is from CUDA managed memory, the NVSHMEM heap, or inside of another registered buffer region. NVSHMEMX_ERROR_OUT_OF_MEMORY for errors in allocating control structures. NVSHMEMX_ERROR_INTERNAL for internal registration errors.

### **NVSHMEMX_BUFFER_UNREGISTER**

`int nvshmemx_buffer_unregister ( void *addr )`

_addr [IN]_
    The Address to be unregistered. Note: This must be the same address used when calling nvshmemx_buffer_register. Passing a value in the middle of a registered buffer will result in the buffer not being unregistered.

**Description**

The _nvshmemx_buffer_unregister_ operation unregisters a buffer previously registered with _nvshmem_buffer_register_.

**Returns**

0 on success. NVSHMEMX_ERROR_INVALID_VALUE if the buffer could not be found.

### **NVSHMEMX_BUFFER_UNREGISTER_ALL**

`void nvshmemx_buffer_unregister_all ( void )`

**Description**

The _nvshmemx_buffer_unregister_all_ operation unregisters all buffers previously registered with _nvshmemx_buffer_register_.

**Returns**

None.

### **NVSHMEMX_BUFFER_REGISTER_SYMMETRIC**

`void * nvshmemx_buffer_register_symmetric ( void *user_buffer , size_t size , int flags )`

_user_buffer [IN]_
    Pointer to memory buffer to be registered.
_size [IN]_
    Size of the buffer.
_flags [IN]_
    Flags to be passed into the API call, reserved for future use. By default, set this to 0.

**Description**

The _nvshmemx_buffer_register_symmetric_ function registers the user allocated buffer as part of the NVSHMEM symmetric heap memory. Once buffers are registered using this function, they can be used as operands for NVSHMEM communication operations like any memory allocated using nvshmem_malloc or nvshmem_calloc functions. This function allows users to “bring-your-own-memory” buffer for communication and replaces the need for using nvshmem_malloc, nvshmem_calloc functions. Memory buffers allocated on both device and Extended GPU memory (EGM) can be registered using this function. For platforms that support NVLINK SHARP, this API would automatically bind device or host side buffer to an existing CUDA multicast group belonging to one or more NVSHMEM teams in this API. As the memory buffer is mapped onto symmetric heap, this routine should be called in all PEs (as it is a collective API call) with buffers of same size. Undefined behavior will be observed if the same size is not passed by all PEs.

**Requirements**

>   * Only memory buffers allocated using CUDA VMM APIs can be registered using this function.
>   * This feature is supported on platforms which support CUDA Virtual Memory Management (VMM) feature.
>   * Buffer physical memory size is multiple of heap memory granularity used in NVSHMEM which is the maximum of
>

>
> (CUMEM recommended granularity obtained using cuMemGetAllocationGranularity(…, CU_MEM_ALLOC_GRANULARITY_RECOMMENED),
>     NVSHMEM_CUMEM_GRANULARITY NVSHMEM environment variable).

**Returns**

If successful, the routine return pointer to the symmetric address where the buffer is mapped, otherwise it returns null pointer.

### **NVSHMEMX_BUFFER_REGISTER_SYMMETRIC_AT_PREFERRED_ADDRESS**

`void * nvshmemx_buffer_register_symmetric_at_preferred_address ( void *user_buffer , size_t size , void* preferred_addr , int flags )`

_user_buffer [IN]_
    Pointer to memory buffer to be registered.
_size [IN]_
    Size of the buffer.
_preferred_addr [IN]_
    Preferred address within symmetric heap where the user provided buffer needs to be mapped.
_flags [IN]_
    Flags to be passed into the API call, reserved for future use. By default, set this to 0.

**Description**

The _nvshmemx_buffer_register_symmetric_at_preferred_address_ function tries to register the user allocated buffer at preferred_addr (if valid and available) within the NVSHMEM symmetric heap memory. In the event the supplied preferred_addr is unavailable then the user allocated buffer will be mapped within the symmetric heap (similar to _nvshmemx_buffer_register_symmetric_). As the memory buffer is mapped onto symmetric heap, this routine should be called in all PEs (as it is a collective API call) with buffers of same size. Undefined behavior will be observed if the same size is not passed by all PEs.

**Requirements**

  * Only memory buffers allocated using CUDA VMM APIs can be registered using this function.
  * This feature is supported on platforms which support CUDA Virtual Memory Management (VMM) feature.
  * Buffer physical memory size is multiple of heap memory granularity used in NVSHMEM which is the maximum of

(CUMEM recommended granularity obtained using cuMemGetAllocationGranularity(…, CU_MEM_ALLOC_GRANULARITY_RECOMMENED),
    NVSHMEM_CUMEM_GRANULARITY NVSHMEM environment variable).

**Returns**

If successful, the routine return pointer to the symmetric address where the buffer is mapped, otherwise it returns null pointer.

### **NVSHMEMX_BUFFER_UNREGISTER_SYMMETRIC**

`int nvshmemx_buffer_unregister_symmetric ( void *ptr , size_t size )`

_ptr [IN]_
    Symmetric address of a registered memory buffer.
_size [IN]_
    Size of the buffer.

**Description**

The _nvshmemx_buffer_unregister_symmetric_ routine unmaps the user allocated buffer from NVSHMEM symmetric heap memory.

**Returns**

0 on success. NVSHMEMX_ERROR_INVALID_VALUE if the arguments passed are not valid symmetric address and size of registered memory buffer. NVSHMEMX_ERROR_INTERNAL for internal errors.
