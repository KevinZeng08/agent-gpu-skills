# NVSHMEM Device Collectives with Numba-CUDA DSL

This section documents the NVSHMEM Device Collective operations with Numba-CUDA DSL.

## Example: Using barrier and broadcast in a Numba-CUDA kernel

The following example demonstrates how to use the NVSHMEM barrier and broadcast collective operations in a Numba-CUDA kernel. Collectives enable synchronization and data movement among multiple PEs (processing elements) directly from device code.

    import numpy as np
    import cupy as cp
    from numba import cuda
    import nvshmem
    import nvshmem.core.device.numba as nvshmem_numba
    from mpi4py import MPI

    @cuda.jit
    def collective_kernel(buf, root, pe):
        tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        # Synchronize all PEs at this point
        nvshmem_numba.barrier_all()
        if tid == 0:
            # Broadcast buf[0] from root PE to all PEs
            nvshmem_numba.broadcast(buf, buf, root)

    # Initialize NVSHMEM
    dev = cudaDevice()
    dev.set_current()
    stream = dev.create_stream()
    nvshmem.init(dev=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi", stream=stream)

    # Get information about the current PE
    me = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    # Set up buffer and root
    buf = nvshmem.array((1,), dtype=np.int32)
    if me == 0:
        buf[0] = 42  # Root PE initializes the value
    root = 0

    # Launch kernel to perform collectives
    # Note, Numba-cuda does not accept a cuda.core Stream, so we need to pass the stream handle.
    collective_kernel[1, 1](buf, root, me, stream=int(stream.handle))

    # Finalize NVSHMEM
    nvshmem.finalize(dev=dev, stream=stream)

This example synchronizes all PEs with barrier_all and then broadcasts the value in buf[0] from the root PE (PE 0) to all PEs. Only thread 0 performs the broadcast for demonstration purposes. In practice, you can have multiple threads participate as needed.

For more details, see the Numba test suite and the NVSHMEM4Py documentation.

`nvshmem.core.device.numba.collective. sync_block ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> )`

Executes a block-wide sync across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.

`nvshmem.core.device.numba.collective. sync_warp ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> )`

Executes a warp-wide sync across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.

`nvshmem.core.device.numba.collective. sync ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> )`

Executes a sync across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.

`nvshmem.core.device.numba.collective. sync_all ( )`

Executes a thread-wide sync across all PEs in the runtime.

`nvshmem.core.device.numba.collective. sync_all_block ( )`

Executes a block-wide sync across all PEs in the runtime.

`nvshmem.core.device.numba.collective. sync_all_warp ( )`

Executes a warp-wide sync across all PEs in the runtime.

`nvshmem.core.device.numba.collective. barrier ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> )`

Executes a thread-wide barrier across the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.

`nvshmem.core.device.numba.collective. barrier_block ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> )`

Executes a block-wide barrier across all threads in the block.

Args:

  * `team` (`Teams`): NVSHMEM team handle.

`nvshmem.core.device.numba.collective. barrier_warp ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> )`

Executes a warp-wide barrier across all threads in the warp.

Args:

  * `team` (`Teams`): NVSHMEM team handle.

`nvshmem.core.device.numba.collective. barrier_all ( )`

Executes a barrier across all PEs in the runtime.

`nvshmem.core.device.numba.collective. barrier_all_block ( )`

Executes a block-wide barrier across all PEs in the runtime.

`nvshmem.core.device.numba.collective. barrier_all_warp ( )`

Executes a warp-wide barrier across all PEs in the runtime.

`nvshmem.core.device.numba.collective. reduce ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , dst_array: numba.core.types.npytypes.Array , src_array: numba.core.types.npytypes.Array , op: str )`

Performs a reduction from src_array to dst_array across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.
  * `dst_array` (`Array`): Symmetric array to store the result of the reduction routine.
  * `src_array` (`Array`): Symmetric array that contains at least one element for each separate reduction routine.
  * `op` (`str`): String representing the reduction operator.

Supported reduction operators: See https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html?highlight=allreduce#nvshmem-reductions for supported reduction operators.

Note: In case that `src_array` and `dst_array` are of different sizes, the size of the smaller array is reduced.

`nvshmem.core.device.numba.collective. reduce_block ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , dst_array: numba.core.types.npytypes.Array , src_array: numba.core.types.npytypes.Array , op: str )`

Performs a block-wide reduction from src_array to dst_array across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.
  * `dst_array` (`Array`): Symmetric array to store the result of the reduction routine.
  * `src_array` (`Array`): Symmetric array that contains the source data.
  * `op` (`str`): String representing the reduction operator.

Supported reduction operators: See https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html?highlight=allreduce#nvshmem-reductions for supported reduction operators.

Note: In case that `src_array` and `dst_array` are of different sizes, the size of the smaller array is reduced.

`nvshmem.core.device.numba.collective. reduce_warp ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , dst_array: numba.core.types.npytypes.Array , src_array: numba.core.types.npytypes.Array , op: str )`

Performs a warp-wide reduction from src_array to dst_array across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.
  * `dst_array` (`Array`): Symmetric array to store the result of the reduction routine.
  * `src_array` (`Array`): Symmetric array that contains the source data.
  * `op` (`str`): String representing the reduction operator.

Supported reduction operators: See https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html?highlight=allreduce#nvshmem-reductions for supported reduction operators.

Note: In case that `src_array` and `dst_array` are of different sizes, the size of the smaller array is reduced.

`nvshmem.core.device.numba.collective. reducescatter ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , dst_array: numba.core.types.npytypes.Array , src_array: numba.core.types.npytypes.Array , op: str )`

Performs a thread-scoped reduce-scatter operation from src_array to dst_array across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.
  * `dst_array` (`Array`): Symmetric array to store the result of the reduce-scatter routine.
  * `src_array` (`Array`): Symmetric array that contains the source data.
  * `op` (`str`): String representing the reduction operator.

Supported reduction operators: See https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html?highlight=allreduce#nvshmem-reductions for supported reduction operators.

Note: Array size is taken from the `dst_array`. The `dst_array` must be >= `(src_array.size // team_n_pes(team))`.

`nvshmem.core.device.numba.collective. reducescatter_block ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , dst_array: numba.core.types.npytypes.Array , src_array: numba.core.types.npytypes.Array , op: str )`

Performs a block-scoped reduce-scatter operation from src_array to dst_array across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.
  * `dst_array` (`Array`): Symmetric array to store the result of the reduce-scatter routine.
  * `src_array` (`Array`): Symmetric array that contains the source data.
  * `op` (`str`): String representing the reduction operator.

Supported reduction operators: See https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html?highlight=allreduce#nvshmem-reductions for supported reduction operators.

Note: Array size is taken from the `dst_array`. The `src_array` must be >= `(dst_array.size * team_n_pes(team))`.

`nvshmem.core.device.numba.collective. reducescatter_warp ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , dst_array: numba.core.types.npytypes.Array , src_array: numba.core.types.npytypes.Array , op: str )`

Performs a warp-scoped reduce-scatter operation from src_array to dst_array across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.
  * `dst_array` (`Array`): Symmetric array to store the result of the reduce-scatter routine.
  * `src_array` (`Array`): Symmetric array that contains the source data.
  * `op` (`str`): String representing the reduction operator.

Supported reduction operators: See https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html?highlight=allreduce#nvshmem-reductions for supported reduction operators.

Note: Array size is taken from the `dst_array`. The `src_array` must be >= `(src_array.size // team_n_pes(team))`.

`nvshmem.core.device.numba.collective. fcollect ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , dst_array: numba.core.types.npytypes.Array , src_array: numba.core.types.npytypes.Array )`

Performs a thread-scoped fcollect operation from src_array to dst_array across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.
  * `dst_array` (`Array`): Symmetric array to store the result of the fcollect routine.
  * `src_array` (`Array`): Symmetric array that contains the source data.

Note: Array size is taken from the `src_array`. The `src_array` must be >= `(src_array.size * team_n_pes(team))`.

`nvshmem.core.device.numba.collective. fcollect_block ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , dst_array: numba.core.types.npytypes.Array , src_array: numba.core.types.npytypes.Array )`

Performs a block-scoped fcollect operation from src_array to dst_array across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.
  * `dst_array` (`Array`): Symmetric array to store the result of the fcollect routine.
  * `src_array` (`Array`): Symmetric array that contains the source data.

Note: Array size is taken from the `src_array`. The `dst_array` must be >= `(src_array.size * team_n_pes(team))`.

`nvshmem.core.device.numba.collective. fcollect_warp ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , dst_array: numba.core.types.npytypes.Array , src_array: numba.core.types.npytypes.Array )`

Performs a warp-scoped fcollect operation from src_array to dst_array across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.
  * `dst_array` (`Array`): Symmetric array to store the result of the fcollect routine.
  * `src_array` (`Array`): Symmetric array that contains the source data.

Note: Array size is taken from the `src_array`. The `dst_array` must be >= `(src_array.size * team_n_pes(team))`.

`nvshmem.core.device.numba.collective. broadcast ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , dst_array: numba.core.types.npytypes.Array , src_array: numba.core.types.npytypes.Array , root: int = 0 )`

Performs a thread-scoped broadcast operation from src_array to dst_array across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.
  * `dst_array` (`Array`): Destination symmetric array.
  * `src_array` (`Array`): Source symmetric array.
  * `root` (`int`): Root PE for the broadcast.

`nvshmem.core.device.numba.collective. broadcast_block ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , dst_array: numba.core.types.npytypes.Array , src_array: numba.core.types.npytypes.Array , root: int = 0 )`

Performs a block-scoped broadcast operation from src_array to dst_array across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.
  * `dst_array` (`Array`): Destination symmetric array.
  * `src_array` (`Array`): Source symmetric array.
  * `root` (`int`): Root PE for the broadcast.

`nvshmem.core.device.numba.collective. broadcast_warp ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , dst_array: numba.core.types.npytypes.Array , src_array: numba.core.types.npytypes.Array , root: int = 0 )`

Performs a warp-scoped broadcast operation from src_array to dst_array across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.
  * `dst_array` (`Array`): Destination symmetric array.
  * `src_array` (`Array`): Source symmetric array.
  * `root` (`int`): Root PE for the broadcast.

`nvshmem.core.device.numba.collective. alltoall ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , dst_array: numba.core.types.npytypes.Array , src_array: numba.core.types.npytypes.Array )`

Performs a thread-scoped alltoall operation from src_array to dst_array across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.
  * `dst_array` (`Array`): Destination symmetric array.
  * `src_array` (`Array`): Source symmetric array.

`nvshmem.core.device.numba.collective. alltoall_block ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , dst_array: numba.core.types.npytypes.Array , src_array: numba.core.types.npytypes.Array )`

Performs a block-scoped alltoall operation from src_array to dst_array across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.
  * `dst_array` (`Array`): Destination symmetric array.
  * `src_array` (`Array`): Source symmetric array.

`nvshmem.core.device.numba.collective. alltoall_warp ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , dst_array: numba.core.types.npytypes.Array , src_array: numba.core.types.npytypes.Array )`

Performs a warp-scoped alltoall operation from src_array to dst_array across all PEs in the team.

Args:

  * `team` (`Teams`): NVSHMEM team handle.
  * `dst_array` (`Array`): Destination symmetric array.
  * `src_array` (`Array`): Source symmetric array.
