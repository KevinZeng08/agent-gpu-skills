# Overview of the APIs

NVSHMEM includes the OpenSHMEM 1.3 API, and several APIs that are defined in OpenSHMEM versions 1.4 and 1.5.

Here are the naming conventions:

>   * OpenSHMEM function names are prefixed with nv, for example, `nvshmem_init`.
>   * Type names are prefixed with nv, for example, `nvshmem_team_t`.
>   * Constants are prefixed with NV, for example, `NVSHMEM_VENDOR_STRING`.
>   * Environment variables are prefixed with `NVSHMEM`, for example, `NVSHMEM_SYMMETRIC_SIZE`.
>

NVSHMEM functions are classified based on where they can be invoked:

>   * On the host.
>   * On the GPU.
>   * On both the host and the GPU.
>

The following APIs can be invoked only on the host:

>   * Initialization and termination, for example, `nvshmem_init` and `nvshmem_finalize`.
>   * Memory management, for example, `nvshmem_malloc` and `nvshmem_free`.
>   * Collective kernel launch, for example, `nvshmemx_collective_launch`.
>   * Stream-based operations.
>

The following APIs can be invoked only on the GPU:

>   * Thread block scoped operations, for example, `nvshmem_putmem_block`.
>   * Thread warp scoped operations, for example, `nvshmem_putmem_warp`.
>

The remaining operations, including one-sided remote memory access, one-sided remote atomic memory access, memory ordering, point-to-point synchronization, collectives, pointer query, and PE information query operations are supported from both the host and the device.

## Unsupported OpenSHMEM 1.3 APIs

The following OpenSHMEM 1.3 APIs are not currently supported in NVSHMEM:

>   * OpenSHMEM Fortran API
>   * `shmem_pe_accessible`
>   * `shmem_addr_accessible`
>   * `shmem_realloc`
>   * `shmem_collect`
>   * `shmem_alltoalls`
>   * `shmem_lock`
>   * `_SHMEM_*` constants (deprecated)
>   * `SMA_*` environment variables (deprecated)
>   * CUDA supports only a subset of the atomic operations in the OpenSHMEM specification. In NVSHMEM, the `long long`, `int64_t`, and `ptrdiff_t` types are currently unsupported for `add`, `fetch_add`, `inc`, and `fetch_inc` atomic operations.
>

## OpenSHMEM 1.3 APIs Not Supported Over Remote Network Transports

The following OpenSHMEM 1.3 APIs are not currently supported over remote network transports in NVSHMEM:

>   * `shmem_iput`
>   * `shmem_iget`
>

## Supported OpenSHMEM APIs (OpenSHMEM 1.4 and 1.5)

The following OpenSHMEM 1.5 APIs are supported by NVSHMEM:

>   * `nvshmem_wait_until_{any, all, some}` and `nvshmem_wait_until_{any, all, some}_vector`
>   * `nvshmem_test_{any, all, some}` and `nvshmem_wait_until_{any, all, some}_vector`
>

The following OpenSHMEM 1.4 APIs are supported by NVSHMEM:

>   * Threading support, with functions `nvshmem_init_thread` and `nvshmem_query_thread`, and `NVSHMEM_THREAD_SINGLE`, `NVSHMEM_THREAD_FUNNELED`, `NVSHMEM_THREAD_SERIALIZED`, and `NVSHMEM_THREAD_MULTIPLE` constants.
>   * `NVSHMEM_SYNC_SIZE` constant
>   * `nvshmem_calloc` (host)
>   * Bitwise atomic memory operations `and`, `fetch_and`, `or`, `fetch_or`, `xor`, and `fetch_xor` (host and device)
>   * `nvshmem_sync` and `nvshmem_sync_all` (host and device)
>

## NVSHMEM API Extensions For CPU Threads

NVSHMEM extension APIs invoked only by CPU threads:

Initialization
    `nvshmemx_init_attr` `nvshmemx_hostlib_init_attr`
CUDA kernel launch

`nvshmemx_collective_launch`

CUDA kernels that invoke synchronizing NVSHMEM APIs such as `nvshmem_barrier`, `nvshmem_wait`, collective operations, and others, must be launched using this API; otherwise behavior is undefined.

Collective launch grid size

`nvshmemx_collective_launch_query_gridsize`

Used to query the largest grid size that can be used for the given kernel with CUDA cooperative launch on the current GPU.

Remote memory access

`nvshmemx_put_<all_variants>_on_stream`, `nvshmemx_get_<all_variants>_on_stream`

Asynchronous with respect to the calling CPU thread; takes a cudaStream_t as argument and is ordered on that CUDA stream.

Memory ordering
    `nvshmemx_quiet_on_stream`, `nvshmemx_flush_on_stream`
Collective communication
    `nvshmemx_<TYPENAME>_broadcast_on_stream`, `nvshmemx_broadcastmem_on_stream`, `nvshmemx_<TYPENAME>_fcollect_on_stream`, `nvshmemx_fcollectmem_on_stream`, `nvshmemx_<TYPENAME>_alltoall_on_stream`, `nvshmemx_alltoallmem_on_stream`, `nvshmemx_<TYPENAME>_<OP>_reduce_on_stream`, and `nvshmemx_<TYPENAME>_<OP>_reducescatter_on_stream`
Collective synchronization
    `nvshmemx_barrier_all_on_stream`, `nvshmemx_barrier_on_stream`, `nvshmemx_sync_all_on_stream`, and `nvshmemx_team_sync_on_stream`

NVSHMEM extends the remote memory access (get and put), memory ordering, collective communication, and collective synchronization APIs with support for CUDA streams. Each stream-based function for an OpenSHMEM operation performs the same operation as described in the OpenSHMEM specification. An additional argument of `cudaStream_t` type is added as the last argument to each function and indicates the stream on which the operation is enqueued.

Ordering APIs (fence, quiet, and barrier) that are issued on the CPU and the GPU only order communication operations that were issued from the CPU and the GPU, respectively. To ensure completion of GPU-side operations from the CPU, the developer must perform a GPU-side quiet operation and ensure completion of the CUDA kernel from which the GPU-side operations were issued, using operations like `cudaStreamSynchronize` or `cudaDeviceSynchronize`. Alternatively, a stream-based quiet operation can be used. Stream-based quiet operations have the effect of a quiet being executed on the GPU in stream order, ensuring completion and ordering of only GPU-side operations.

`nvshmemx_flush_on_stream` can be used when only source-buffer reuse is required. It does not guarantee remote visibility; use `nvshmemx_quiet_on_stream` before a remote consumer depends on the destination data being visible.

## NVSHMEM API Extensions For GPU Threads

RMA Write

`nvshmemx_put_block`, `nvshmemx_put_warp`

New APIs for GPU-side invocation are provided that can be called collectively by a threadblock or a warp.

RMA Read
    `nvshmemx_get_block`, `nvshmemx_get_warp`
Asynchronous RMA write
    `nvshmemx_put_nbi_block`, `nvshmemx_put_nbi_warp`
Asynchronous RMA read
    `nvshmemx_get_nbi_block`, `nvshmemx_get_nbi_warp`
Memory ordering

`nvshmemx_flush`, `nvshmemx_flush_block`, `nvshmemx_flush_warp`

`nvshmemx_flush` is a single-thread device routine. If a different GPU thread issues the operations being flushed, GPU thread synchronization is required before calling `nvshmemx_flush`. The `_block` and `_warp` variants are collective calls that include this synchronization for the participating block or warp. Flush can be used when only source-buffer reuse is required; it does not guarantee remote visibility.

Collective communication
    `nvshmemx_<TYPENAME>_broadcast_block`, `nvshmemx_<TYPENAME>_broadcast_warp`, `nvshmemx_broadcastmem_block`, `nvshmemx_broadcastmem_warp`, `nvshmemx_<TYPENAME>_fcollect_block`, `nvshmemx_<TYPENAME>_fcollect_warp`, `nvshmemx_fcollectmem_block`, `nvshmemx_fcollectmem_warp`, `nvshmemx_<TYPENAME>_alltoall_block`, `nvshmemx_<TYPENAME>_alltoall_warp`, `nvshmemx_alltoallmem_block`, `nvshmemx_alltoallmem_warp`, `nvshmemx_<TYPENAME>_<OP>_reduce_block`, `nvshmemx_<TYPENAME>_<OP>_reduce_warp`, `nvshmemx_<TYPENAME>_<OP>_reducescatter_block`, and `nvshmemx_<TYPENAME>_<OP>_reducescatter_warp`
Collective synchronization
    `nvshmemx_barrier_all_block`, `nvshmemx_barrier_all_warp`, `nvshmemx_barrier_block`, `nvshmemx_barrier_warp`, `nvshmemx_sync_all_block`, `nvshmemx_sync_all_warp`, `nvshmemx_team_sync_block`, and `nvshmemx_team_sync_warp`

These extension APIs can be invoked by GPU threads. Most of these APIs have two variants each: one with the `_block` suffix and the other with the `_warp` suffix. For example, the OpenSHMEM API `shmem_float_put` has two extension APIs in NVSHMEM, `nvshmemx_float_put_block` and `nvshmemx_float_put_warp`.

The `_block` and `_warp` extension APIs are collective calls that must be called by every thread in the scope of the API and with exactly the same arguments. The scope of the `*_block` extension APIs is the block in which the thread resides. Similarly, the scope of the `*_warp` extension API is the warp in which the thread resides. For example, if thread 0 calls `nvshmemx_float_put_block`, then every other thread that is in the same block as thread 0 must also call `nvshmemx_float_put_block` with the same arguments. Otherwise, the call results in erroneous behavior or a deadlock in the program. The NVSHMEM runtime might or might not leverage the multiple threads in the scope of the API to execute the API call.

The extension APIs are useful in the following situations:

>   * Converting `nvshmem_float_put` to `nvshmemx_float_put_block` enables the NVSHMEM runtime to leverage all the threads in the block to concurrently copy the data to the destination PE if the destination GPU of the put call is p2p connected. If the destination GPU is connected via a remote network, then a single thread in the block can issue an RMA write operation to the destination GPU.
>   * The `*_block` and `*_warp` extensions of the collective APIs can use multiple threads to perform collective operations, such as parallel reduction operations of multiple threads sending data in parallel.
>

## Tile-Granular Collective APIs

To facilitate development of computation-communication fused kernels, NVSHMEM supports device-side Tile-granular collective APIs which perform collective operations on individual tiles of data instead of entire buffer. Note: these APIs are based on CUDA C++ and not part of cuTile programming model.

Input and output operands for these APIs are tiles which are represented using `nvshmemx::Tensor`, an abstraction for a multi-dimensional array (similar to CuTe Tensor). `nvshmemx::Tensor` is composed of data pointer (starting address of first tensor element) and `nvshmemx::Layout` (similar to CuTe Layout). `nvshmemx::Layout` indicates the structure of the tensor using shape (tuple indicating the number of elements in each dimension) and stride (tuple of stride along each dimension). Each tile could be processed by a thread, warp, warpgroup or threadblock and we support these scopes for performing the collectives. In order to ensure collectives can be performed independently across tiles, each concurrently executing tile-collective should use a unique NVSHMEM team. Tile collectives performed iteratively can re-use teams across iterations.

Restrictions:

  * Tile-granular collective APIs introduced are currently experimental. We welcome the community to use the APIs but these APIs may undergo changes to improve user experience.
  * Tile-granular collectives are only supported on NVLink SHARP based systems currently and support up to 2D tensors.
  * APIs currently support `float`, `half`, `cutlass::half_t`, `__nv_bfloat16`, and `cutlass::bfloat16_t` data types. Using CUTLASS datatypes requires building with `NVSHMEM_BUILD_WITH_CUTLASS` cmake flag set. `CUTLASS_HOME` environment variable must be set to the path of the CUTLASS directory.
