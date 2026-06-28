# Collective Communication

_Collective routines_ are defined as coordinated communication or synchronization operations performed by a group of PEs.

NVSHMEM provides the following types of collective routines:

  1. Collective routines that operate on teams use a team handle parameter to determine which PEs will participate in the routine, and use resources encapsulated by the team object to perform operations. See Section Team Management for details on team management.
  2. Collective routines that accept no team parameters implicitly operate on all PEs.

Concurrent accesses to symmetric memory by an NVSHMEM collective routine and any other means of access—where at least one updates the symmetric memory—results in undefined behavior. Since PEs can enter and exit collectives at different times, accessing such memory remotely may require additional synchronization.

## Team-based collectives

The team-based collective routines are performed with respect to a valid NVSHMEM team, which is specified by a team handle argument. Team-based collective operations require all PEs in the team to call the routine in order for the operation to complete. If an invalid team handle or `NVSHMEM_TEAM_INVALID` is passed to a team-based collective routine, the behavior is undefined.

All NVSHMEM teams-based collective operations are blocking routines. On return from a team-based collective call, the PE may immediately call another collective operation on that same team. Team-based collectives must occur in the same program order across all PEs in a team.

While NVSHMEM routines provide thread support according to the thread-support level provided at initialization (see Section Thread Support), team-based collective routines may not be called simultaneously by multiple threads on a given team.

The team-based collective routines defined in this NVSHMEM Implementation are:

  * `nvshmem_team_sync`
  * `nvshmem_[TYPENAME_]alltoall`
  * `nvshmem_[TYPENAME_]broadcast`
  * `nvshmem_[TYPENAME_]fcollect`
  * `nvshmem_[TYPENAME_]{and, or, xor, max, min, sum, prod}_reduce`
  * `nvshmemx_[TYPENAME_]{min, max, sum}_reducescatter`

In addition, all team creation functions are collective operations. In addition to the ordering and thread safety requirements described here, there are additional synchronization requirements on team creation operations. See Section Team Management for more details.

## Implicit team collectives

Some NVSHMEM collective routines implicitly operate on all PEs. These routines include:

  * `nvshmem_sync_all`, which synchronizes all PEs in the computation.
  * `nvshmem_barrier_all`, which synchronizes all PEs in the computation and ensures completion of all local and remote memory updates.
  * NVSHMEM memory-management routines, which imply one or more calls to a routine equivalent to `nvshmem_barrier_all`.

## Tile-based Collectives

These collectives operate on individual tiles instead of the contiguous data buffers. NVSHMEM provides constructs along with helper functions to create and manage tensors. These constructs and helper functions can be used in both host and device code. However, the tile-based collective APIs are supported only on the device. All tile-granular APIs and helper functions are part of _nvshmemx_ namespace.

### Tile helper functions

**Shape** : Tuple containing the size of tensor along each dimension.

`template <class... Ts> __host__ __device__ constexpr shape<Ts...> make_shape ( Ts const&... t )`

**Stride** : Tuple containing stride along each dimension.

`template <class... Ts> __host__ __device__ constexpr stride<Ts...> make_stride ( Ts const&... t )`

_make_shape()_ and _make_stride()_ functions take in a variadic number of arguments indicating the size and stride along each dimension and returns a shape and stride tuple respectively. The APIs support the use of _cuda::std::integral_constant_ static constants as arguments. It is recommended to use static constants for shape and stride wherever possible. Users may use `ConstInt<N>` as a shorthand for _cuda::std::integral_constant <int, N>_.

**Layout** : Tuple of shape and stride used to describe the structure of a tile.

`template <class Shape, class Stride> __host__ __device__ constexpr Layout<Shape, Stride> make_layout ( Shape const& shape , Stride const& stride )`

**Tensor** : Struct to represent a tile, composed of address of starting element and layout. The data buffer backing a tensor should be part of NVSHMEM symmetric memory for the tensor to be used for tile collectives.

`template <typename T, class Layout> __host__ __device__ constexpr Tensor ( T* data , Layout layout )`

Tile collectives are team-based and are performed with respect to a valid NVSHMEM team as specified by the handle. To ensure tile collectives can make independent progress, concurrently executing tile collectives should use unique NVSHMEM teams. The input data buffer is split into multiple tiles. Performing AllReduce (or Reduce, AllGather) collective on entire data buffer is equivalent to performing an AllReduce (or Reduce, AllGather) collective on each of the underlying tile. The Tile-granular APIs currently supported are:

  * `tile_{max, min, sum}_reduce`
  * `tile_{max, min, sum}_reduce_{warp, warpgroup, block}`
  * `tile_{max, min, sum}_rooted_reduce`
  * `tile_{max, min, sum}_rooted_reduce_{warp, warpgroup, block}`
  * `tile_allgather`
  * `tile_allgather_{warp, warpgroup, block}`
  * `tile_collective_wait`

### Tile collective algorithms

The users are expected to specify the algorithm to be used for each collective. The list of algorithms supported is shown below:

  * `tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI`
  * `tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI`
  * `tile_coll_algo_t::NVLS_TWO_SHOT_PUSH_NBI`

The algorithms containing “ _NBI_ ” in their name indicate non-blocking collectives. The users are expected to use _tile_collective_wait_ routine to ensure completion of the non-blocking collectives. NVLink SHARP-based algorithms can have PUSH or PULL-based implementations as indicated in the name. PUSH denotes that data is moved from source PE tensor to destination PEs while PULL indicates getting data from destination PEs to locate PE. Two-shot NVLS algorithm demonstrates both PUSH and PULL behavior. For PULL-based collectives, the collective routines will internally perform a fence and synchronize across PEs within the team to ensure data from all participating PEs are ready before pulling them in.

## Error codes returned from team-based collectives

Collective operations involving multiple PEs may return values indicating success while other PEs are still executing the collective operation. Return values indicating success or failure of a collective routine on one PE may not indicate that all PEs involved in the collective operation will return the same value. Some operations, such as team creation, must return identical return codes across multiple PEs.

## Collective operations scopes and active sets

Collective operations on a given team (including the implicit team operating on all PEs) must be called by all PEs in the team concurrently. However, collective operations operating on distinct teams may be concurrent as long as the user can guarantee forward progress (e.g. by using collective launch APIs).

Collective operations must be called by exactly one instance per PEs at a time. This means device APIs must be called by exactly one thread (for regular thread scoped APIs), warp (for warp scoped APIs) or block (for block scoped APIs) at a time. In addition, warp (resp. block) scoped APIs must be called with identical arguments by all CUDA threads in the warp (resp. block).

Note that, warp (resp. block) scoped APIs can called by the any warp (resp. block) within the CUDA grid, where warps are identified by their block and lane ID, and blocks are identified by their block ID.

This, for instance, is well-defined because we have only 1 active block at a time.

    __global__ void kernel() {
        if(blockIdx.x == nvshmem_my_pe()) {
            nvshmemx_barrier_all_block();
        }
    }

    gridDims grid(2, 1, 1);
    blockDims(32, 1, 1);
    nvshmemx_collective_launch(kernel, grid, block, ...); // On 2 PEs

ote that, this is ill-defined because two blocks are concurrently calling the barrier on the same team (TEAM_WORLD), even with a collective launch API

    void kernel() {
        nvshmemx_barrier_all_block();
    }

    gridDims grid(2, 1, 1);
    blockDims(32, 1, 1);
    nvshmemx_collective_launch(kernel, grid, block, ...); // On 2 PEs

he following, on the other-hand, is well defined. Because we can guarantee that the blocks will be co-scheduled (using the collective launch), each block will make forward progress. In addition, each block is running a collective on a different team, which is allowed.

    void kernel() {
        nvshmemx_barrier_block(teams[blockIdx.x]);
    }

    gridDims grid(2, 1, 1);
    blockDims(32, 1, 1);
    nvshmemx_collective_launch(kernel, grid, block, ...); // On 2 PEs

**NVSHMEM_BARRIER_ALL**

`void nvshmem_barrier_all ( void )`

`void nvshmemx_barrier_all_on_stream ( void, cudaStream_t stream )`

`__device__ void nvshmem_barrier_all ( void )`

`__device__ void nvshmemx_barrier_all_block ( void )`

`__device__ void nvshmemx_barrier_all_warp ( void )`

**Description**

The `nvshmem_barrier_all` routine is a mechanism for synchronizing all PEs at once. This routine blocks the calling PE until all PEs have called `nvshmem_barrier_all`. In a multithreaded NVSHMEM program, only the calling thread is blocked, however, it may not be called concurrently by multiple threads in the same PE.

Prior to synchronizing with other PEs, `nvshmem_barrier_all` ensures completion of all previously issued memory stores and remote memory updates issued NVSHMEMAMOs and RMA routine calls such as `nvshmem_int_add`, `nvshmem_put32`, `nvshmem_put_nbi`, and `nvshmem_get_nbi`.

**Returns**

None.

**Notes**

The `nvshmem_barrier_all` routine can be used to portably ensure that memory access operations observe remote updates in the order enforced by initiator PEs.

Ordering APIs (`nvshmem_fence`, `nvshmem_quiet`, `nvshmem_barrier`, and `nvshmem_barrier_all`) issued on the CPU and the GPU only order communication operations that were issued from the CPU and the GPU, respectively. To ensure completion of GPU-side operations from the CPU, the developer must perform a GPU-side quiet operation and ensure completion of the CUDA kernel from which the GPU-side operations were issued, using operations like `cudaStreamSynchronize` or `cudaDeviceSynchronize`. Alternatively, a stream-based quiet operation can be used. Stream-based quiet operations have the effect of a quiet being executed on the GPU in stream order, ensuring completion and ordering of only GPU-side operations.

The following `nvshmem_barrier_all` example is for _C_ programs: ./example_code/shmem_barrierall_example.c

## **NVSHMEM_BARRIER**

`int nvshmem_barrier ( nvshmem_team_t team )`

`int nvshmemx_barrier_on_stream ( nvshmem_team_t team , cudaStream_t stream )`

`__device__ int nvshmem_barrier ( nvshmem_team_t team )`

`__device__ int nvshmemx_barrier_block ( nvshmem_team_t team )`

`__device__ int nvshmemx_barrier_warp ( nvshmem_team_t team )`

_team [IN]_
    The team over which to perform the barrier.

**Description**

`nvshmem_barrier` is a collective synchronization routine over a team. Control returns from `nvshmem_barrier` after all PEs in the team have called `nvshmem_barrier`.

As with all NVSHMEM collective routines, each of these routines assumes that all PEs in the team call the routine. If a PE outside the team calls this collective routine, the behavior is undefined.

`nvshmem_barrier` ensures that all previously issued stores and remote memory updates, including AMOs and RMA operations, done by any of the PEs in the team are complete before returning. On systems with only NVLink, updates are globally visible, whereas on systems with both NVLink and InfiniBand, NVSHMEM only guarantees that updates to the memory of a given PE are visible to that PE.

**Returns**

Zero on successful local completion. Nonzero otherwise.

**Notes**

The `nvshmem_barrier` routine can be used to portably ensure that memory access operations observe remote updates in the order enforced by initiator PEs.

Ordering APIs (`nvshmem_fence`, `nvshmem_quiet`, `nvshmem_barrier`, and `nvshmem_barrier_all`) issued on the CPU and the GPU only order communication operations that were issued from the CPU and the GPU, respectively. To ensure completion of GPU-side operations from the CPU, the developer must perform a GPU-side quiet operation and ensure completion of the CUDA kernel from which the GPU-side operations were issued, using operations like `cudaStreamSynchronize` or `cudaDeviceSynchronize`. Alternatively, a stream-based quiet operation can be used. Stream-based quiet operations have the effect of a quiet being executed on the GPU in stream order, ensuring completion and ordering of only GPU-side operations.

The following barrier example is for _C_ programs: ./example_code/shmem_barrier_example.c

## **NVSHMEM_SYNC**

`int nvshmem_sync ( nvshmem_team_t team )`

`int nvshmemx_sync_on_stream ( nvshmem_team_t team , cudaStream_t stream )`

`__device__ int nvshmem_sync ( nvshmem_team_t team )`

`__device__ int nvshmemx_sync_block ( nvshmem_team_t team )`

`__device__ int nvshmemx_sync_warp ( nvshmem_team_t team )`

`int nvshmem_team_sync ( nvshmem_team_t team )`

`int nvshmemx_team_sync_on_stream ( nvshmem_team_t team , cudaStream_t stream )`

`__device__ int nvshmem_team_sync ( nvshmem_team_t team )`

`__device__ int nvshmemx_team_sync_block ( nvshmem_team_t team )`

`__device__ int nvshmemx_team_sync_warp ( nvshmem_team_t team )`

_team [IN]_
    The team over which to perform the operation.

**Description**

`nvshmem_sync` is a collective synchronization routine over a team.

The routine registers the arrival of a PE at a synchronization point in the program. This is a fast mechanism for synchronizing all PEs that participate in this collective call. The routine blocks the calling PE until all PEs in the specified team have called `nvshmem_sync`. In a multithreaded NVSHMEM program, only the calling thread is blocked.

Team-based sync routines operate over all PEs in the provided team argument. All PEs in the provided team must participate in the sync operation. If `team` compares equal to `NVSHMEM_TEAM_INVALID` or is otherwise invalid, the behavior is undefined.

In contrast with the `nvshmem_barrier` routine, `nvshmem_sync` only ensures completion and visibility of previously issued memory stores and does not ensure completion of remote memory updates issued via NVSHMEM routines.

**Returns**

Zero on successful local completion. Nonzero otherwise.

**Notes**

The `nvshmem_sync` routine can be used to portably ensure that memory access operations observe remote updates in the order enforced by the initiator PEs, provided that the initiator PE ensures completion of remote updates with a call to `nvshmem_quiet` prior to the call to the `nvshmem_sync` routine.

The following `nvshmem_sync` example is for _C_ programs: ./example_code/shmem_sync_example.c

## **NVSHMEM_SYNC_ALL**

`void nvshmem_sync_all ( void )`

`void nvshmemx_sync_all_on_stream ( void, cudaStream_t stream )`

`__device__ void nvshmem_sync_all ( void )`

`__device__ void nvshmemx_sync_all_block ( void )`

`__device__ void nvshmemx_sync_all_warp ( void )`

**Description**

This routine blocks the calling PE until all PEs in the world team have called `nvshmem_sync_all`.

In a multithreaded NVSHMEM program, only the calling thread is blocked.

In contrast with the `nvshmem_barrier_all` routine, `nvshmem_sync_all` only ensures completion and visibility of previously issued memory stores and does not ensure completion of remote memory updates issued via NVSHMEM routines.

**Returns**

None.

**Notes**

The `nvshmem_sync_all` routine is equivalent to calling `nvshmem_team_sync` on the world team.

## **NVSHMEM_ALLTOALL**

`int nvshmem_TYPENAME_alltoall ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nelems )`

`int nvshmemx_TYPENAME_alltoall_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nelems , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_alltoall ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nelems )`

`__device__ int nvshmemx_TYPENAME_alltoall_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nelems )`

`__device__ int nvshmemx_TYPENAME_alltoall_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nelems )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

`int nvshmem_alltoallmem ( shmem_team_t team , void *dest , const void *source , size_t nelems )`

`int nvshmemx_alltoallmem_on_stream ( shmem_team_t team , void *dest , const void *source , size_t nelems , cudaStream_t stream )`

`__device__ int nvshmem_alltoallmem ( shmem_team_t team , void *dest , const void *source , size_t nelems )`

`__device__ int nvshmemx_alltoallmem_block ( shmem_team_t team , void *dest , const void *source , size_t nelems )`

`__device__ int nvshmemx_alltoallmem_warp ( shmem_team_t team , void *dest , const void *source , size_t nelems )`

_team [IN]_
    A valid NVSHMEM team handle to a team.
_dest [OUT]_
    Symmetric address of a data object large enough to receive the combined total of `nelems` elements from each PE in the team. The type of `dest` should match that implied in the SYNOPSIS section.
_source [IN]_
    Symmetric address of a data object that contains `nelems` elements of data for each PE in the team, ordered according to destination PE. The type of `source` should match that implied in the SYNOPSIS section.
_nelems [IN]_
    The number of elements to exchange for each PE. For `nvshmem_alltoallmem`, elements are bytes; for `nvshmem_alltoall{32,64}`, elements are 4 or 8 bytes, respectively.

**Description**

The `nvshmem_alltoall` routines are collective routines. Each PE participating in the operation exchanges `nelems` data elements with all other PEs participating in the operation.

The data being sent and received are stored in a contiguous symmetric data object. The total size of each PE’s `source` object and `dest` object is `nelems` times the size of an element times `N`, where `N` equals the number of PEs participating in the operation. The `source` object contains `N` blocks of data (where the size of each block is defined by `nelems`) and each block of data is sent to a different PE.

The same `dest` and `source` arrays, and same value for nelems must be passed by all PEs that participate in the collective.

Given a PE `i` that is the \\({\textit{k}^{\text{\tiny th}}}\\) PE participating in the operation and a PE `j` that is the \\({\textit{l}^{\text{\tiny th}}}\\) PE participating in the operation,

PE `i` sends the \\({\textit{l}^{\text{\tiny th}}}\\) block of its `source` object to the \\({\textit{k}^{\text{\tiny th}}}\\) block of the `dest` object of PE `j`.

Team-based collect routines operate over all PEs in the provided team argument. All PEs in the provided team must participate in the collective. If `team` compares equal to `NVSHMEM_TEAM_INVALID` or is otherwise invalid, the behavior is undefined.

Before any PE calls a `nvshmem_alltoall` routine, the following conditions must be ensured:

  * The `dest` data object on all PEs in the active set is ready to accept the `nvshmem_alltoall` data.

Otherwise, the behavior is undefined.

Upon return from a `nvshmem_alltoall` routine, the following is true for the local PE:

  * Its `dest` symmetric data object is completely updated and the data has been copied out of the `source` data object.

**Returns**

Zero on successful local completion. Nonzero otherwise.

This _C/C++_ example shows a `nvshmem_int64_alltoall` on two 64-bit integers among all PEs. ./example_code/shmem_alltoall_example.c

## **NVSHMEM_BROADCAST**

`int nvshmem_TYPENAME_broadcast ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nelems , int PE_root )`

`int nvshmemx_TYPENAME_broadcast_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nelems , int PE_root , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_broadcast ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nelems , int PE_root )`

`__device__ int nvshmemx_TYPENAME_broadcast_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nelems , int PE_root )`

`__device__ int nvshmemx_TYPENAME_broadcast_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nelems , int PE_root )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

`int nvshmem_broadcastmem ( shmem_team_t team , void *dest , const void *source , size_t nelems , int PE_root )`

`int nvshmemx_broadcastmem_on_stream ( shmem_team_t team , void *dest , const void *source , size_t nelems , int PE_root , cudaStream_t stream )`

`__device__ int nvshmem_broadcastmem ( shmem_team_t team , void *dest , const void *source , size_t nelems , int PE_root )`

`__device__ int nvshmemx_broadcastmem_block ( shmem_team_t team , void *dest , const void *source , size_t nelems , int PE_root )`

`__device__ int nvshmemx_broadcastmem_warp ( shmem_team_t team , void *dest , const void *source , size_t nelems , int PE_root )`

_team [IN]_
    The team over which to perform the operation.
_dest [OUT]_
    Symmetric address of destination data object. The type of `dest` should match that implied in the SYNOPSIS section.
_source [IN]_
    Symmetric address of the source data object. The type of `source` should match that implied in the SYNOPSIS section.
_nelems [IN]_
    The number of elements in `source` and `dest` arrays. For `nvshmem_broadcastmem`, elements are bytes; for `nvshmem_broadcast{32,64}`, elements are 4 or 8 bytes, respectively.
_PE_root [IN]_
    Zero-based ordinal of the PE, with respect to the team, from which the data is copied.

**Description**

NVSHMEM broadcast routines are collective routines over a team. They copy the `source` data object on the PE specified by `PE_root` to the `dest` data object on the PEs participating in the collective operation. The same `dest` and `source` data objects and the same values of `PE_root` and `nelems` must be passed by all PEs participating in the collective operation.

For team-based broadcasts:

  * The `dest` object is updated on all PEs.
  * All PEs in the `team` argument must participate in the operation.
  * If `team` compares equal to `NVSHMEM_TEAM_INVALID` or is otherwise invalid, the behavior is undefined.
  * PE numbering is relative to the team. The specified root PE must be a valid PE number for the team, between `0` and `N` \\(-\\)`1`, where `N` is the size of the team.

Before any PE calls a broadcast routine, the following conditions must be ensured:

  * The `dest` array on all PEs participating in the broadcast is ready to accept the broadcast data.

Otherwise, the behavior is undefined.

Upon return from a broadcast routine, the following are true for the local PE:

  * For team-based broadcasts, the `dest` data object is updated.
  * The `source` data object may be safely reused.

**Returns**

For team-based broadcasts, zero on successful local completion; otherwise, nonzero.

**Notes**

Team handle error checking and integer return codes are currently undefined. Implementations may define these behaviors as needed, but programs should ensure portability by doing their own checks for invalid team handles and for `NVSHMEM_TEAM_INVALID`.

In the following _C_ example, the call to `nvshmem_broadcast` copies `source` on PE \\(0\\) to `dest` on PEs \\(0\dots npes-1\\).

_C/C++_ example:

./example_code/shmem_broadcast_example.c

## **NVSHMEM_FCOLLECT**

`int nvshmem_TYPENAME_fcollect ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nelems )`

`int nvshmemx_TYPENAME_fcollect_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nelems , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_fcollect ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nelems )`

`__device__ int nvshmemx_TYPENAME_fcollect_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nelems )`

`__device__ int nvshmemx_TYPENAME_fcollect_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nelems )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

`int nvshmem_fcollectmem ( shmem_team_t team , void *dest , const void *source , size_t nelems )`

`int nvshmemx_fcollectmem_on_stream ( shmem_team_t team , void *dest , const void *source , size_t nelems , cudaStream_t stream )`

`__device__ int nvshmem_fcollectmem ( shmem_team_t team , void *dest , const void *source , size_t nelems )`

`__device__ int nvshmemx_fcollectmem_block ( shmem_team_t team , void *dest , const void *source , size_t nelems )`

`__device__ int nvshmemx_fcollectmem_warp ( shmem_team_t team , void *dest , const void *source , size_t nelems )`

_team [IN]_
    A valid NVSHMEM team handle.
_dest [OUT]_
    Symmetric address of an array large enough to accept the concatenation of the `source` arrays on all participating PEs. The type of `dest` should match that implied in the SYNOPSIS section.
_source [IN]_
    Symmetric address of the source data object. The type of `source` should match that implied in the SYNOPSIS section.
_nelems [IN]_
    The number of elements in `source` array. The fcollect routines require that nelems be the same value in all participating PEs.

**Description**

NVSHMEM `fcollect` routines perform a collective operation to concatenate `nelems` data items from the `source` array into the `dest` array, over a team in processor number order. The resultant `dest` array contains the contribution from PEs as follows:

  * For a team, the data from PE number `0` in the team is first, then the contribution from PE `1` in the team, and so on.

The collected result is written to the `dest` array for all PEs that participate in the operation. The same `dest` and `source` arrays must be passed by all PEs that participate in the operation.

Team-based collect routines operate over all PEs in the provided team argument. All PEs in the provided team must participate in the operation. If `team` compares equal to `NVSHMEM_TEAM_INVALID` or is otherwise invalid, the behavior is undefined.

Upon return from a collective routine, the following are true for the local PE:

  * The `dest` array is updated and the `source` array may be safely reused.

**Returns**

Zero on successful local completion. Nonzero otherwise.

The following `nvshmem_collect` example is for _C/C++_ programs: ./example_code/shmem_collect_example.c

## **NVSHMEM_REDUCTIONS**

Types, Names, and Operations for Team-Based Reductions _TYPE_ | _TYPENAME_ | Operations Supporting _TYPE_ |  |
---|---|---|---|---
char | char |  | MAX, MIN | SUM, PROD
signed char | schar |  | MAX, MIN | SUM, PROD
short | short |  | MAX, MIN | SUM, PROD
int | int |  | MAX, MIN | SUM, PROD
long | long |  | MAX, MIN | SUM, PROD
long long | longlong |  | MAX, MIN | SUM, PROD
ptrdiff_t | ptrdiff |  | MAX, MIN | SUM, PROD
unsigned char | uchar | AND, OR, XOR | MAX, MIN | SUM, PROD
unsigned short | ushort | AND, OR, XOR | MAX, MIN | SUM, PROD
unsigned int | uint | AND, OR, XOR | MAX, MIN | SUM, PROD
unsigned long | ulong | AND, OR, XOR | MAX, MIN | SUM, PROD
unsigned long long | ulonglong | AND, OR, XOR | MAX, MIN | SUM, PROD
int8_t | int8 | AND, OR, XOR | MAX, MIN | SUM, PROD
int16_t | int16 | AND, OR, XOR | MAX, MIN | SUM, PROD
int32_t | int32 | AND, OR, XOR | MAX, MIN | SUM, PROD
int64_t | int64 | AND, OR, XOR | MAX, MIN | SUM, PROD
uint8_t | uint8 | AND, OR, XOR | MAX, MIN | SUM, PROD
uint16_t | uint16 | AND, OR, XOR | MAX, MIN | SUM, PROD
uint32_t | uint32 | AND, OR, XOR | MAX, MIN | SUM, PROD
uint64_t | uint64 | AND, OR, XOR | MAX, MIN | SUM, PROD
size_t | size | AND, OR, XOR | MAX, MIN | SUM, PROD
float | float |  | MAX, MIN | SUM, PROD
half | half |  | MAX, MIN | SUM, PROD
__nv_bfloat16 | bfloat16 |  | MAX, MIN | SUM, PROD
double | double |  | MAX, MIN | SUM, PROD

### AND

Performs a bitwise AND reduction across a set of PEs.

`int nvshmem_TYPENAME_and_reduce ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`int nvshmemx_TYPENAME_and_reduce_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_and_reduce ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_and_reduce_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_and_reduce_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

where _TYPE_ is one of the integer types supported for the AND operation as specified by Table teamreducetypes.

`int nvshmem_TYPENAME_and_reducescatter ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`int nvshmemx_TYPENAME_and_reducescatter_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_and_reducescatter ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_and_reducescatter_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_and_reducescatter_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

where _TYPE_ is one of the integer types supported for the AND operation as specified by Table teamreducetypes.

### OR

Performs a bitwise OR reduction across a set of PEs.

`int nvshmem_TYPENAME_or_reduce ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`int nvshmemx_TYPENAME_or_reduce_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_or_reduce ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_or_reduce_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_or_reduce_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

where _TYPE_ is one of the integer types supported for the OR operation as specified by Table teamreducetypes.

`int nvshmem_TYPENAME_or_reducescatter ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`int nvshmemx_TYPENAME_or_reducescatter_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_or_reducescatter ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_or_reducescatter_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_or_reducescatter_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

where _TYPE_ is one of the integer types supported for the OR operation as specified by Table teamreducetypes.

### XOR

Performs a bitwise exclusive OR (XOR) reduction across a set of PEs.

`int nvshmem_TYPENAME_xor_reduce ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`int nvshmemx_TYPENAME_xor_reduce_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_xor_reduce ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_xor_reduce_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_xor_reduce_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

where _TYPE_ is one of the integer types supported for the XOR operation as specified by Table teamreducetypes.

`int nvshmem_TYPENAME_xor_reducescatter ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`int nvshmemx_TYPENAME_xor_reducescatter_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_xor_reducescatter ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_xor_reducescatter_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_xor_reducescatter_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

where _TYPE_ is one of the integer types supported for the XOR operation as specified by Table teamreducetypes.

### MAX

Performs a maximum-value all or local reduction across a set of PEs.

`int nvshmem_TYPENAME_max_reduce ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`int nvshmemx_TYPENAME_max_reduce_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_max_reduce ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_max_reduce_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_max_reduce_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

where _TYPE_ is one of the integer or real types supported for the MAX operation as specified by Table teamreducetypes.

`int nvshmem_TYPENAME_max_reducescatter ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`int nvshmemx_TYPENAME_max_reducescatter_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_max_reducescatter ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_max_reducescatter_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_max_reducescatter_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

where _TYPE_ is one of the integer or real types supported for the MAX operation as specified by Table teamreducetypes.

### MIN

Performs a minimum-value all or local reduction across a set of PEs.

`int nvshmem_TYPENAME_min_reduce ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`int nvshmemx_TYPENAME_min_reduce_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_min_reduce ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_min_reduce_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_min_reduce_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

where _TYPE_ is one of the integer or real types supported for the MIN operation as specified by Table teamreducetypes.

`int nvshmem_TYPENAME_min_reducescatter ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`int nvshmemx_TYPENAME_min_reducescatter_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_min_reducescatter ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_min_reducescatter_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_min_reducescatter_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

where _TYPE_ is one of the integer or real types supported for the MIN operation as specified by Table teamreducetypes.

### SUM

Performs a sum all or local reduction across a set of PEs.

`int nvshmem_TYPENAME_sum_reduce ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`int nvshmemx_TYPENAME_sum_reduce_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_sum_reduce ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_sum_reduce_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_sum_reduce_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

where _TYPE_ is one of the integer, real, or complex types supported for the SUM operation as specified by Table teamreducetypes.

`int nvshmem_TYPENAME_sum_reducescatter ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`int nvshmemx_TYPENAME_sum_reducescatter_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_sum_reducescatter ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_sum_reducescatter_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_sum_reducescatter_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

where _TYPE_ is one of the integer, real, or complex types supported for the SUM operation as specified by Table teamreducetypes.

### PROD

Performs a product reduction across a set of PEs.

`int nvshmem_TYPENAME_prod_reduce ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`int nvshmemx_TYPENAME_prod_reduce_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_prod_reduce ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_prod_reduce_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_prod_reduce_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

where _TYPE_ is one of the integer, real, or complex types supported for the PROD operation as specified by Table teamreducetypes.

`int nvshmem_TYPENAME_prod_reducescatter ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`int nvshmemx_TYPENAME_prod_reducescatter_on_stream ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce , cudaStream_t stream )`

`__device__ int nvshmem_TYPENAME_prod_reducescatter ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_prod_reducescatter_block ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

`__device__ int nvshmemx_TYPENAME_prod_reducescatter_warp ( nvshmem_team_t team , TYPE *dest , const TYPE *source , size_t nreduce )`

where _TYPE_ is one of the integer, real, or complex types supported for the PROD operation as specified by Table teamreducetypes.

_team [IN]_
    The team over which to perform the operation.
_dest [OUT]_
    Symmetric address of an array, of length `nreduce` elements, to receive the result of the all or local reduction routines. The type of `dest` should match that implied in the SYNOPSIS section.
_source [IN]_
    Symmetric address of an array, of length `nreduce` elements, that contains one element for each separate reduction routine. The type of `source` should match that implied in the SYNOPSIS section.
_nreduce [IN]_
    The number of elements in the `dest` and `source` arrays.

**Description**

NVSHMEM reduction routines are collective routines over an active set that compute one or more all or local reductions across symmetric arrays on multiple PEs. A reduction performs an associative binary routine across a set of values.

The `nreduce` argument determines the number of separate reductions to perform. The `source` array on all PEs participating in the reduction provides one element for each reduction. The results of the reductions are placed in the `dest` array on all PEs participating in the reduction. A local reduction will only update local PEs with its contribution, while all reduction will perform an additional exchange of each PEs contribution such that end result would be identical on all PEs.

The `source` and `dest` arguments must either be the same symmetric address, or two different symmetric addresses corresponding to buffers that do not overlap in memory. That is, they must be completely overlapping or completely disjoint. In-place reductions are not supported for specific algorithms out-of-the-box.

Team-based reduction routines operate over all PEs in the provided team argument. All PEs in the provided team must participate in the reduction. If `team` compares equal to `NVSHMEM_TEAM_INVALID` or is otherwise invalid, the behavior is undefined.

The value of argument `nreduce` must be equal on all PEs in the active set.

Before any PE calls a reduction routine, the following conditions must be ensured:

  * The `dest` array on all PEs participating in the reduction is ready to accept the results of the _reduction_.

Otherwise, the behavior is undefined.

Upon return from a reduction routine, the following are true for the local PE:

  * The `dest` array is updated and the `source` array may be safely reused.

The complex-typed interfaces are only provided for sum and product reductions. When the _C_ translation environment does not support complex types [1], an NVSHMEM implementation is not required to provide support for these complex-typed interfaces.

**Returns**

Zero on successful local completion. Nonzero otherwise.

In the following _C_ example, each PE intializes an array of random integers with values between \\(0\\) and \\(npes-1\\), inclusively. An OR reduction then tracks the array indices where maximal values occur (maximal values equal \\(npes - 1\\)), and a SUM reduction counts the total number of maximal values across all PEs. ./example_code/shmem_reduce_example.c

## **TILE_REDUCTIONS**

**MAX**

Performs a maximum-value reduction across a set of PE.

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_max_reduce ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_max_reduce_warp ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_max_reduce_warpgroup ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_max_reduce_block ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_max_rooted_reduce ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_max_rooted_reduce_warp ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_max_rooted_reduce_warpgroup ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_max_rooted_reduce_block ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

**MIN**

Performs a minimum-value reduction across a set of PE.

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_min_reduce ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_min_reduce_warp ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_min_reduce_warpgroup ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_min_reduce_block ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_min_rooted_reduce ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_min_rooted_reduce_warp ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_min_rooted_reduce_warpgroup ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_min_rooted_reduce_block ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

**SUM**

Performs a sum reduction across a set of PE.

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_sum_reduce ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_sum_reduce_warp ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_sum_reduce_warpgroup ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_sum_reduce_block ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_sum_rooted_reduce ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_sum_rooted_reduce_warp ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_sum_rooted_reduce_warpgroup ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_sum_rooted_reduce_block ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int root , uint64_t flags )`

_team [IN]_
    The team over which to perform the operation.
_src [IN]_
    Tensor corresponding to the source data tile for the collective.
_dst [OUT]_
    Tensor corresponding to the destination data tile of the collective.
_start_coord [IN]_
    Tuple (of type tuple_t) containing the coordinate of the starting element of the src tensor.
_boundary [IN]_
    Tuple (of type tuple_t) containing the actual problem size of the multi-dimensional array / matrix of the src tensor.
_root [IN]_
    PE which should perform the reduction of the src tile.
_flags [IN]_
    Flags to be passed into the API call. Currently unused and should be set to 0.
**Description**

Reduce routines perform AllReduce collective across PEs while rooted_reduce routines perform Reduce on ‘root’ PE. The reduction routines (reduce and rooted_reduce) operate on individual tiles of data. src and dst tensors are of type src_tensor_t and dst_tensor_t respectively. These tensors are created using Tile helper functions (make_layout(), Tensor()) and contain their layout and datatype information.

Rooted reduce currently supports the tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI algorithm, while reduce routine (AllReduce collective) supports both tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI and tile_coll_algo_t::NVLS_TWO_SHOT_PUSH_NBI algorithms. These APIs support float, half, and __nv_bfloat16 data types. MIN and MAX reduction operations are only supported on half and __nv_bfloat16 datatypes.

Users should ensure that when a PE calls a collective routine, its local src tensor is ready with input data and all dst tensors on all participating PEs are ready to receive data.

If problem sizes do not match the exact tile size, out-of-bounds (OOB) accesses can be predicated using start_coord and boundary. The size of the start_coord and boundary tuples should be equal to the number of dimensions in the src tensor. The make_shape() helper function can be used to create these tuples.

The root argument is required for rooted Reduce and two-shot AllReduce collectives. The two-shot algorithm performs AllReduce as a rooted Reduce followed by a Broadcast, where reduction of the tile is performed by the specified root PE. For two-shot AllReduce, root should ideally be set such that reduction of different tiles is distributed evenly across PEs. This argument is ignored for one-shot tile Reduce collectives.

**Returns**
    On success, the function returns zero; otherwise, a non-zero error code is returned.

## **TILE_ALLGATHER**

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_allgather ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_allgather_warp ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_allgather_warpgroup ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_allgather_block ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , uint64_t flags )`

_team [IN]_
    The team over which to perform the operation.
_src [IN]_
    Tensor corresponding to the source data tile for the collective.
_dst [OUT]_
    Tensor corresponding to the destination data tile of the collective.
_start_coord [IN]_
    Tuple (of type tuple_t) containing the coordinate of the starting element of the src tensor.
_boundary [IN]_
    Tuple (of type tuple_t) containing the actual problem size of the multi-dimensional array / matrix of the src tensor.
_flags [IN]_
    Flags to be passed into the API call. Currently unused and should be set to 0.

**Description**

> This function performs allgather operation on individual tiles of data across all acp{PE} within the team. src and dst tensors are of type src_tensor_t and dst_tensor_t respectively. These tensors are created using Tile helper functions (make_layout(), Tensor()) and contain their layout and datatype information. This API supports float, half, cutlass::half_t, __nv_bfloat16, and cutlass::bfloat16_t data types currently and can uses tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI algorithm.
>
> Users should ensure that when a PE calls a collective routine, its local src tensor is ready with input data and all dst tensors on all participating PE are ready to receive data.
>
> If problem sizes are not exact tile size. Out-of-bounds (OOB) accesses can be predicated using start_coord and boundary. The size of start_coord and boundary tuples should be equal to number of dimensions in src tensor. make_shape() helper function can be used to create this tuples.

**Returns**
    On success, the function returns zero; otherwise, a non-zero error code is returned.

## **TILE_BROADCAST**

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_broadcast ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_broadcast_warp ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_broadcast_warpgroup ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_coll_algo_t algo > __device__ int tile_broadcast_block ( nvshmem_team_t team , src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , uint64_t flags )`

_team [IN]_
    The team over which to perform the operation.
_src [IN]_
    Tensor corresponding to the source data tile for the collective.
_dst [OUT]_
    Tensor corresponding to the destination data tile of the collective.
_start_coord [IN]_
    Tuple (of type tuple_t) containing the coordinate of the starting element of the src tensor.
_boundary [IN]_
    Tuple (of type tuple_t) containing the actual problem size of the multi-dimensional array / matrix of the src tensor.
_flags [IN]_
    Flags to be passed into the API call. Currently unused and should be set to 0.

**Description**

> This function performs a broadcast operation on an individual tile of data across all PE within the team. The PE where the API is called broadcasts its src tensor to the dst tensors of all PEs in the team. Both tensors are of type src_tensor_t and dst_tensor_t respectively, created using Tile helper functions (make_layout(), Tensor()), which contain layout and datatype information.
>
> Currently supported data types are float, half, cutlass::half_t, __nv_bfloat16, and cutlass::bfloat16_t. The algorithm used is tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI.
>
> Users should ensure that its local src tensor is ready with input data and all dst tensors on all participating PE are ready to receive data.
>
> When the problem size does not exactly match the tile size, out-of-bounds (OOB) accesses can be predicated using start_coord and boundary. The size of these tuples must equal the number of dimensions in the src tensor. Use the make_shape() helper function to create these tuples.

**Returns**
    On success, the function returns zero; otherwise, a non-zero error code is returned.

## **TILE_WAIT**

`template < nvshmemx::tile_coll_algo_t algo > __device__ int tile_collective_wait ( nvshmem_team_t team , uint64_t flags )`

_team [IN]_
    The team over which to perform the operation.
_flags [IN]_
    Flags to be passed into the API call. Currently unused and should be set to 0.

**Description**

> If non-blocking collectives algorithms (algorithms containing NBI in their name) are specified for a collective call, tile_collective_wait routine should be called by all participating PEs to ensure the collective is complete. This routine will ensure all previously issued collectives on the specified team, using the specified algorithm are finished and the output tensor is ready to be consumed on the calling PE. Note that completion of this routine on a particular ac{PE} does not guarantee output being ready on other PE within the team.

**Returns**
    On success, the function returns zero; otherwise, a non-zero error code is returned.
[1]| That is, under _C_ language standards prior to _C_ or under _C_ when `__STDC_NO_COMPLEX__` is defined to 1
---|---
