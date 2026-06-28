# Memory Ordering

The following section discusses NVSHMEMAPIs that provide mechanisms to ensure ordering and/or delivery of completion on memory store, blocking, and nonblocking NVSHMEM routines. Table [mem-order] lists the operations affected by NVSHMEM memory ordering routines.

Operations affected by NVSHMEM Memory Ordering routines Operations | Fence | Quiet
---|---|---
Memory Store | X | X
Blocking _Put_ | X | X
Blocking _Get_ |  |
Blocking _AMO_ | X | X
Non-blocking _Put_ | X | X
Non-blocking _Get_ |  | X
Non-blocking _AMO_ | X [1] | X

## **NVSHMEM_FENCE**

`void nvshmem_fence ( void )`

`__device__ void nvshmem_fence ( void )`

**Description**

This routine ensures ordering of delivery of operations on symmetric data objects. Table [mem-order] lists the operations that are ordered by the `nvshmem_fence` routine. All operations on symmetric data objects issued to a particular PE prior to the call to `nvshmem_fence` are guaranteed to be delivered before any subsequent operations on symmetric data objects to the same PE. `nvshmem_fence` guarantees order of delivery, not completion. It does not guarantee order of delivery of nonblocking _Get_ or values fetched by nonblocking AMO routines.

Fence operations issued on the CPU and the GPU only order communication operations that were issued from the CPU and the GPU, respectively.

**Returns**

None.

**Notes**

`nvshmem_fence` only provides per-PE ordering guarantees and does not guarantee completion of delivery. `nvshmem_fence` also does not have an effect on the ordering between memory accesses issued by the target PE. `nvshmem_wait_until`, `nvshmem_test`, `nvshmem_barrier`, `nvshmem_barrier_all` routines can be called by the target PE to guarantee ordering of its memory accesses. There is a subtle difference between `nvshmem_fence` and `nvshmem_quiet`, in that, `nvshmem_quiet` guarantees completion of all operations on symmetric data objects which makes the updates visible to all other PEs.

The `nvshmem_quiet` routine should be called if completion of operations on symmetric data objects is desired when multiple PEs are involved.

In an NVSHMEM program with multithreaded PEs, it is the user’s responsibility to ensure ordering between operations issued by the threads in a PE that target symmetric memory and calls by threads in that PE to `nvshmem_fence`. The `nvshmem_fence` routine can enforce memory store ordering only for the calling thread. Thus, to ensure ordering for memory stores performed by a thread that is not the thread calling `nvshmem_fence`, the update must be made visible to the calling thread according to the rules of the memory model associated with the threading environment.

In device code, making an operation visible to the thread that calls `nvshmem_fence` means ensuring that the operation has been issued before the fence is called; it does not mean that the remote update is complete. For example, after a block-scoped _Put_ operation returns and all threads in the CTA synchronize with `__syncthreads`, a single thread in the CTA may call `nvshmem_fence` before issuing a signal to order the signal after the preceding _Put_ operation. Not every issuing thread needs to call `nvshmem_fence` in this pattern.

`nvshmem_fence` is sufficient when the program only needs to order a later operation after earlier operations to the same PE. If the program needs those earlier operations to be complete or visible before it continues, use `nvshmem_quiet`.

The following example uses `nvshmem_fence` in a _C_ program: ./example_code/shmem_fence_example.c `Put1` will be ordered to be delivered before `put3` and `put2` will be ordered to be delivered before `put4`.

See Ring Broadcast Example for example usage of `nvshmem_fence`.

## **NVSHMEM_QUIET**

`void nvshmem_quiet ( void )`

`__device__ void nvshmem_quiet ( void )`

`void nvshmemx_quiet_on_stream ( cudaStream_t stream )`

**Description**

The `nvshmem_quiet` routine ensures completion of all operations on symmetric data objects issued by the calling PE. Table [mem-order] lists the operations for which the `nvshmem_quiet` routine ensures completion. `nvshmem_quiet` is a local, non-collective operation. Each PE may call it independently to complete operations issued by that PE; the call does not synchronize with or notify other PEs. To notify or coordinate with other PEs, use a collective synchronization operation such as `nvshmem_barrier` or `nvshmem_barrier_all`, or a point-to-point synchronization pattern such as a signal operation followed by a wait operation. Visibility is only guaranteed at the destination PE.

Quiet operations issued on the CPU and the GPU only complete communication operations that were issued from the CPU and the GPU, respectively. To ensure completion of GPU-side operations from the CPU, the developer must perform a GPU-side quiet operation and ensure completion of the CUDA kernel from which the GPU-side operations were issued, using operations like `cudaStreamSynchronize` or `cudaDeviceSynchronize`. Alternatively, a stream-based quiet operation can be used. Stream-based quiet operations have the effect of a quiet being executed on the GPU in stream order, ensuring completion and ordering of only GPU-side operations.

**Returns**

None.

**Notes**

`nvshmem_quiet` is most useful as a way of ensuring completion of several operations on symmetric data objects initiated by the calling PE. For example, one might use `nvshmem_quiet` to await delivery of a block of data before issuing another _Put_ or nonblocking _Put_ routine, which sets a completion flag on another PE. `nvshmem_quiet` is not usually needed if `nvshmem_barrier_all` or `nvshmem_barrier` are called. The barrier routines wait for the completion of outstanding operations to symmetric data objects on all PEs.

In an NVSHMEM program with multithreaded PEs, it is the user’s responsibility to ensure ordering between operations issued by the threads in a PE that target symmetric memory and calls by threads in that PE to `nvshmem_quiet`. The `nvshmem_quiet` routine can enforce memory store ordering only for the calling thread. Thus, to ensure ordering for memory stores performed by a thread that is not the thread calling `nvshmem_quiet`, the update must be made visible to the calling thread according to the rules of the memory model associated with the threading environment.

A call to `nvshmem_quiet` by a thread completes the operations posted prior to calling `nvshmem_quiet`. If the user intends to also complete operations issued by a thread that is not the thread calling `nvshmem_quiet`, the user must ensure that the operations are performed prior to the call to `nvshmem_quiet`. This may require the use of a synchronization operation provided by the threading package. For example, when using POSIX Threads, the user may call the `pthread_barrier_wait` routine to ensure that all threads have issued operations before a thread calls `nvshmem_quiet`.

The same rule applies to GPU threads. If multiple threads issue NVSHMEM operations and then synchronize so that those operations have been issued before a single elected thread calls `nvshmem_quiet`, the single `nvshmem_quiet` call completes the preceding operations. For example, after a block-scoped _Put_ operation returns and all threads in the CTA synchronize with `__syncthreads`, one thread may call `nvshmem_quiet` to complete the block-scoped _Put_ operation. Not every issuing thread needs to call `nvshmem_quiet` in this pattern.

`nvshmem_quiet` does not have an effect on the ordering between memory accesses issued by the target PE. `nvshmem_wait_until`, `nvshmem_test`, `nvshmem_barrier`, `nvshmem_barrier_all` routines can be called by the target PE to guarantee ordering of its memory accesses.

The following example uses `nvshmem_quiet` in a _C_ program: ./example_code/shmem_quiet_example.c `Put1` and `put2` will be completed and visible before `put3` and `put4`.

## **NVSHMEMX_FLUSH**

`__device__ void nvshmemx_flush ( void )`

`__device__ void nvshmemx_flush_warp ( void )`

`__device__ void nvshmemx_flush_block ( void )`

`void nvshmemx_flush_on_stream ( cudaStream_t stream )`

_stream [IN]_
    A CUDA stream on which to enqueue the flush operation.

**Description**

The `nvshmemx_flush`, `nvshmemx_flush_warp`, and `nvshmemx_flush_block` routines ensure local completion of outstanding nonblocking _Put_ operations issued by the calling PE. After the flush routine returns, the source buffers used by those operations can be safely reused, overwritten, or freed.

The `nvshmemx_flush_on_stream` routine enqueues a stream-ordered flush on `stream`. The flush completes after the source buffers used by nonblocking _Put_ operations previously enqueued on the same stream are safe to reuse.

Flush does not guarantee that data is visible at the destination PE and does not replace `nvshmem_quiet` or `nvshmem_fence`. A program must still use `nvshmem_quiet`, `nvshmemx_quiet_on_stream`, or another appropriate synchronization operation before a remote consumer depends on the destination data being visible.

Flush is primarily useful when the source buffer can be reused before remote visibility is required, such as NVLink/TMA transfers that use a temporary source buffer. For IB/RoCE paths, flush can require a transport drain similar to `nvshmem_quiet`; use it only when source-buffer reuse is the required completion property.

**Returns**

None.

**Notes**

`nvshmemx_flush` is a single-thread device routine. The `nvshmemx_flush_warp` and `nvshmemx_flush_block` routines are warp-scoped and block-scoped flush primitives, respectively. They are collective GPU-thread routines; every thread in the warp or block, respectively, must call the routine with the same control flow.

A flush can only complete operations that have been issued before the flush call. If one GPU thread issues the _Put_ operation and another GPU thread issues `nvshmemx_flush`, the program must synchronize the issuing threads before the flush. Use `__syncwarp` for warp-scoped ordering or `__syncthreads` for block-scoped ordering, as appropriate. The `nvshmemx_flush_warp` and `nvshmemx_flush_block` routines include this synchronization for the participating warp or block.

The following device code uses a block-scoped nonblocking _Put_ operation and then calls `nvshmemx_flush_block` so the source buffer can be reused after the flush returns.

    #include <nvshmem.h>
    #include <nvshmemx.h>

    __global__ void flush_example(float *dest, const float *source,
                                  size_t nelems, int pe) {
        nvshmemx_float_put_nbi_block(dest, source, nelems, pe);

        nvshmemx_flush_block();

        /* source can be reused by the block here. */
    }

[1]| NVSHMEM fence routines does not guarantee order of delivery of values fetched by nonblocking AMO routines.
---|---
