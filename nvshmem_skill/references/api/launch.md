# Kernel Launch Routines

## **NVSHMEMX_COLLECTIVE_LAUNCH**

`int nvshmemx_collective_launch ( const void *func , dim3 gridDims , dim3 blockDims , void **args , size_t sharedMem , cudaStream_t stream )`

_func [IN]_
    A pointer to the function to launch on the device.
_gridDims [IN]_
    The grid dimensions.
_blockDims [IN]_
    The block dimensions.
_args [IN]_

Arguments to be passed to the device function.

If the kernel has N parameters the args argument should point to array of N pointers. Each pointer, from args[0] to args[N - 1], points to the region of memory from which the actual parameter will be copied.

_sharedMem [IN]_
    Sets the amount of dynamic shared memory that will be available to each thread block.
_stream [IN]_
    The stream on which the kernel should be launched.

**Description**

The nvshmemx_collective_launch function must be used to launch CUDA kernels on the GPU when the CUDA kernels use NVSHMEM synchronization or collective APIs (e.g., nvshmem_wait, nvshmem_barrier, nvshmem_barrier_all, or any other collective operation). CUDA kernels that do not use synchronizing NVSHMEM APIs (or that do not use NVSHMEM APIs at all), are not required to be launched by this API. This call is collective across the PEs in the NVSHMEM job.

This function invokes kernel func on all PEs on gridDim (gridDim.x, gridDim.y, gridDim.z) grid of blocks using CUDA cooperative launch. When gridDim is set to 0, the NVSHMEM runtime picks the largest grid size that can be used for the given kernel with CUDA cooperative launch on the current GPU. Each block contains blockDim (blockDim.x, blockDim.y, blockDim.z) threads.

The device on which this kernel is invoked must have a non-zero value for the device attribute cudaDevAttrCooperativeLaunch.

The total number of blocks launched cannot exceed the maximum number of blocks as returned by nvshmemx_collective_launch_query_gridsize.

**Returns**

Returns 0 on success or an error code on failure.

**Example**

See Collective Launch Example.

## **NVSHMEMX_COLLECTIVE_LAUNCH_QUERY_GRIDSIZE**

`int nvshmemx_collective_launch_query_gridsize ( const void *func , dim3 blockDims , void **args , size_t sharedMem , int *gridsize )`

_func [IN]_
    A pointer to the function to launch on the device.
_blockDims [IN]_
    The block dimensions.
_args [IN]_
    Arguments to be passed to the device function.
_sharedMem [IN]_
    Sets the amount of dynamic shared memory that will be available to each thread block.
_gridsize [OUT]_
    The function returns in this variable the largest number of blocks (grid size) that can be used for collective launch.

**Description**

The nvshmemx_collective_launch_query_gridsize call is used to query the largest grid size that can be used for the kernel func with block dimension blockDims using CUDA cooperative launch on the current GPU. Limiting the number of blocks to gridsize ensures that threads performing NVSHMEM synchronization and collective operations are executed concurrently. The return value of this function can be passed as an argument to nvshmemx_collective_launch.

**Returns**

Returns 0 on success or an error code on failure.
