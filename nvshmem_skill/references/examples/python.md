# NVSHMEM4Py Examples

## UID-Based Initialization Example

The following code shows an example of initializing NVSHMEM using the attribute-based initialization API. It uses an MPI Communicator to perform broadcast of the NVSHMEM attributes to all participating processes.

    """
    This example shows how to initialize NVSHMEM4Py using an NVSHMEM Unique ID.

    In this example, MPI4Py is used to perform a broadcast to share the Unique ID object across all processes.

    To initialize NVSHMEM4Py with a custom launcher, you can use a similar approach, replacing the MPI4Py broadcast with data movement handled by your custom launcher.
    """

    import mpi4py.MPI as MPI
    import nvshmem.core
    import numpy as np
    from cuda.core.experimental import Device, system

    # Use MPI4Py to retrieve the MPI communicator and rank information
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    # Find a Device which is unique to each rank
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
    dev = Device(local_rank_per_node)
    dev.set_current()

    # Create an empty uniqueid for all ranks
    uniqueid = nvshmem.core.get_unique_id(empty=True)
    if rank == 0:
        # Rank 0 gets a populated uniqueid
        uniqueid = nvshmem.core.get_unique_id()

    # Broadcast UID to all ranks
    # This is what the custom launcher would need to do if you want to avoid using MPI4Py
    comm.Bcast(uniqueid._data.view(np.int8), root=0)

    # NVSHMEM Processing Elements (PEs) are bound to a specific device at init time
    nvshmem.core.init(device=dev, uid=uniqueid, rank=rank, nranks=nranks,
                      initializer_method="uid")

    # Do your NVSHMEM work here

    nvshmem.core.finalize()

## MPI Comm-Based Initialization Example

The following code shows an example of initializing NVSHMEM using the MPI init method.

    """
    This example shows how to initialize NVSHMEM4Py using an MPI4Py communicator.
    """

    import mpi4py.MPI as MPI
    import nvshmem.core
    import numpy as np
    from cuda.core.experimental import Device, system

    # Use MPI4Py to retrieve the MPI communicator and rank information
    comm = MPI.COMM_WORLD

    # Find a Device which is unique to each rank
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
    dev = Device(local_rank_per_node)
    dev.set_current()

    # NVSHMEM Processing Elements (PEs) are bound to a specific device at init time
    # If you use `emulated_mpi` instead of MPI, nvshmem4py will internally perform an MPI broadcast using a UniqueID to perform init.
    # This has the advantage of not requiring NVSHMEM and MPI to be compiled for each other. The only requirement is a working MPI4Py environment
    nvshmem.core.init(device=dev, mpi_comm=comm, initializer_method="mpi")

    # Do your NVSHMEM work here

    nvshmem.core.finalize()

## Torch.distributed ProcessGroup Initialization Example

The following code shows two methods of initializing NVSHMEM using the UID bootstrap with the broadcast of the UniqueID handled by a `torch.distributed.ProcessGroup`.

    """
    This file contains two examples of how to initialize NVSHMEM4Py using the UID bootstrap with the broadcast of the UniqueID handled by a ``torch.distributed.ProcessGroup``.

    The first example uses ``torch.distributed.broadcast`` on a NumPy array extracted from the ``nvshmem.core.UniqueID`` object.

    The second example uses ``torch.distributed.broadcast_object_list`` to broadcast the ``nvshmem.core.UniqueID`` object directly.

    You would launch a program using these examples with a command like `torchrun --nproc-per-node <NGPUs> torch_init.py`
    """

    import torch.distributed as dist
    import torch
    import nvshmem.core
    import os
    from cuda.core.experimental import Device, system

    def torchrun_uid_init_bcast():
        """
        Initialize NVSHMEM using UniqueID with `torchrun` as the launcher

        It uses torch.distributed.broadcast on a NumPy array to handle the broadcasting
        """
        # Set Torch device
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        # nvshmem4py requires a cuda.core Device at init time
        dev = Device(local_rank)
        dev.set_current()
        global stream
        stream = dev.create_stream()

        # Initialize torch.distributed process group
        world_size = torch.cuda.device_count()
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=local_rank,
            world_size=world_size,
            device_id=device
        )

        # Extract rank, nranks from process group
        num_ranks = dist.get_world_size()
        rank_id = dist.get_rank()

        # Create an empty uniqueid for all ranks
        uniqueid = nvshmem.core.get_unique_id(empty=True)
        if rank_id == 0:
            # Rank 0 gets a real uniqueid
            uniqueid = nvshmem.core.get_unique_id()

        # This is a NumPy array which is the same shape as a struct nvshmem_uniqueid_t
        data = uniqueid._data
        # We use torch.distributed.broadcast to send the UID to all ranks
        dist.broadcast(data, src=0)
        dist.barrier()

        if rank_id != 0:
            uniqueid._data = data

        nvshmem.core.init(device=dev, uid=uniqueid, rank=rank_id, nranks=num_ranks, initializer_method="uid")

    def torchrun_uid_init_bcast_object():
        """
        Initialize NVSHMEM using UniqueID with `torchrun` as the launcher

        It uses torch.distributed.broadcast_object_list to broadcast the Python Uniqueid
        """
        # Set Torch device
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        # nvshmem4py requires a cuda.core Device at init time
        dev = Device(local_rank)
        dev.set_current()
        global stream
        stream = dev.create_stream()

        # Initialize torch.distributed process group
        world_size = torch.cuda.device_count()
        dist.init_process_group(
            backend="cpu:gloo",
            rank=local_rank,
            world_size=world_size,
            device_id=device
        )

        # Extract rank, nranks from process group
        num_ranks = dist.get_world_size()
        rank_id = dist.get_rank()

        # Create an empty uniqueid for all ranks
        uniqueid = nvshmem.core.get_unique_id(empty=True)
        if rank_id == 0:
            # Rank 0 gets a real uniqueid
            uniqueid = nvshmem.core.get_unique_id()
            broadcast_objects = [uniqueid]
        else:
            broadcast_objects = [None]

        # We use torch.distributed.broadcast_object_list to send the UID to all ranks
        dist.broadcast_object_list(broadcast_objects, src=0)
        dist.barrier()

        nvshmem.core.init(device=dev, uid=broadcast_objects[0], rank=rank_id, nranks=num_ranks, initializer_method="uid")

## Simple P2P Kernel Example

The following code uses Numba-CUDA DSL to express a simple point-to-point communication kernel in Python. It performs a simple ring-based communication between PEs, which each PE writing its ID to its right neighbor’s memory.

    """
    This file shows a minimal example of using NVSHMEM4Py to run a collective operation on CuPy arrays.
    This example demonstrates direct GPU-to-GPU communication using NVSHMEM's symmetric memory model,
    showing how to perform point-to-point operations between NVLink-Accesible PEs (Processing Elements)
    using the nvshmem.core.get_peer_array() function.
    """

    import cupy
    import nvshmem.core
    from cuda.core.experimental import Device, system
    from numba import cuda

    @cuda.jit
    def simple_shift(arr, dst_pe):
        """
        CUDA kernel that performs a simple point-to-point communication.
        Writes the destination PE's ID directly to the array on the target GPU.
        This operation uses NVLink for direct GPU-to-GPU communication if the destination PE is in the same NVLink domain.
        The array passed in should be retrieved using nvshmem.core.get_peer_array() which returns an Array on the symmetric heap that points to another PE's memory.
        """
        # This line is issued as an NVLink Store to the destination PE
        arr[0] = dst_pe

    # Initialize NVSHMEM Using an MPI communicator
    # Calculate local rank within the node to determine which GPU to use
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
    dev = Device(local_rank_per_node)
    dev.set_current()  # Set the current CUDA device
    stream = dev.create_stream()  # Create CUDA stream for asynchronous operations

    # Initialize NVSHMEM with MPI communicator
    nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

    # Create a symmetric array that is accessible from all PEs
    # This array will be used for point-to-point communication
    array = nvshmem.core.array((1,), dtype="int32")

    # Get current PE ID and calculate destination PE for the ring communication
    my_pe = nvshmem.core.my_pe()
    # A unidirectional ring - always get the neighbor to the right
    dst_pe = (my_pe + 1) % nvshmem.core.n_pes()

    # Get a view of the destination PE's array for direct access
    # This enables direct GPU-to-GPU communication over NVLink
    # Note: The destination PE must be in the same NVLink domain
    # If it's not accessible, this will raise an Exception
    dev_dst = nvshmem.core.get_peer_array(b, dst_pe)

    # Launch the CUDA kernel to perform the point-to-point communication
    block = 1
    grid = (size + block - 1) // block
    simple_shift[block, grid, 0, 0](array, my_pe)

    # Synchronize all PEs in the node to ensure communication is complete
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_NODE, stream)

    # Print the result - should show the value written by the neighboring PE
    print(f"From PE {my_pe}, array contains {array}")

    # Clean up NVSHMEM resources
    nvshmem.core.free_array(arr_src)
    nvshmem.core.free_array(arr_dst)
    nvshmem.core.finalize()  # Finalize NVSHMEM

## On-stream Kernels Example

The following code uses Numba-CUDA DSL to express two accumulate kernels in Python. It demonstrates GPU-centric communication using NVSHMEM on CUDA streams, showing how to perform NVLINK-SHARP enabled reductions and collective operations across multiple GPUs.

It is the Python version of the on-stream.cu example.

    """
    This is the Python version of the NVSHMEM on_stream.cu example.
    This example demonstrates distributed GPU computing using NVSHMEM with CUDA streams,
    showing how to perform parallel reductions and collective operations across multiple GPUs.
    """

    import cupy
    from numba import cuda
    from cuda.core.experimental import Device, system

    import nvshmem.core

    from mpi4py import MPI

    # Constants for the correction operation
    THRESHOLD = 42  # Threshold value that triggers correction
    CORRECTION = 7  # Value to subtract when threshold is exceeded

    @cuda.jit
    def accumulate(input, partial_sum):
    	"""
    	Accumulate kernel: Input is a 1-d array and partial_sum is a 1x1 array
    	This kernel performs a parallel reduction to compute the sum of all elements
    	in the input array using atomic operations.
    	"""
    	index = cuda.threadIdx.x
    	if index == 0:
    		partial_sum[0] = 0  # Initialize partial sum to zero
    	cuda.syncthreads()  # Ensure all threads see the initialized value
    	numba.cuda.atomic.add(partial_sum, 0, input[index])  # Atomic addition to partial sum

    @cuda.jit
    def correct_accumulate(input, partial_sum, full_sum):
    	"""
    	Correction kernel: Applies a correction to input values if the full sum exceeds threshold
    	and then recomputes the partial sum.
    	"""
    	index = cuda.threadIdx.x
    	if (full_sum > THRESHOLD):
    		input[index] = input[index] - CORRECTION  # Apply correction if threshold exceeded
    	if index == 0:
    		partial_sum[0] = 0  # Reset partial sum
    	cuda.syncthreads()  # Ensure all threads see the reset value
    	numba.cuda.atomic.add(partial_sum, 0, input[index])

    # Initialize NVSHMEM Using an MPI communicator
    # Calculate local rank within the node to determine which GPU to use
    local_rank_per_node = MPI.COMM_WORLD.Get_rank() % system.num_devices
    dev = Device(local_rank_per_node)
    dev.set_current()

    # Initialize NVSHMEM with MPI communicator for multi-GPU communication
    nvshmem.core.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

    # Get NVSHMEM process information
    mype = nvshmem.core.my_pe()  # Get current PE (Processing Element) ID
    npes = nvshmem.core.n_pes()  # Get total number of PEs
    mype_node = nvshmem.core.team_my_pe(nvshmem.core.Teams.TEAM_NODE)  # Get PE ID within node

    # Define array sizes and create CUDA stream
    input_nelems = 512  # Number of elements in input array
    to_all_elems = 1    # Number of elements for collective operations
    stream = dev.create_stream()  # Create CUDA stream for asynchronous operations

    # Allocate NVSHMEM arrays for distributed memory operations
    input = nvshmem.core.array((input_nelems,), dtype="float32")  # Input array
    partial_sum = nvshmem.core.array((1,), dtype="float32")       # Local partial sum
    full_sum = nvshmem.core.array((1,), dtype="float32")          # Global sum

    # Launch accumulate kernel to compute partial sum
    accumulate[1, input_nelems, 0, stream](input, partial_sum)

    # Perform global reduction to compute full sum across all PEs
    nvshmem.core.reduce(nvshmem.core.Teams.TEAM_WORLD, full_sum, partial_sum, "sum", stream=stream)

    # Launch correction kernel
    correct_accumulate[1, input_nelems, 0, stream](input, partial_sum, full_sum)
    stream.sync()  # Wait for all operations to complete

    print(f"[{mype} of {npes}] Run complete")

    # Clean up NVSHMEM resources
    nvshmem.core.free_array(input)
    nvshmem.core.free_array(partial_sum)
    nvshmem.core.free_array(full_sum)
    nvshmem.core.finalize()

## PyTorch and Triton Interoperability Example

The following code shows how to use NVSHMEM4Py with PyTorch and Triton. It shows how to perform a UID-based initializaiton of NVSHMEM4Py using a Torch.distributed ProcessGroup. It also contains a simple kernel expressed in Triton interoperating with NVSHME4Py host APIs.

    """
    This example initializes NVSHMEM4Py with the `torchrun`
    launcher and torch.distributed

    It runs a kernel expressed with Triton

    Run this program with `torchrun --nproc-per-node <NGPUs> torch_triton_interop.py`
    """

    import torch.distributed as dist
    import torch
    import triton
    import triton.language as tl
    import nvshmem.core
    import os
    from cuda.core.experimental import Device, system

    ###
    #  Helper code from https://github.com/NVIDIA/cuda-python/blob/main/cuda_core/examples/pytorch_example.py
    #  Used to extract PyTorch Stream into a cuda.core.Stream for NVSHMEM APIs
    ###

    # Create a wrapper class that implements __cuda_stream__
    # Example of using https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol
    class PyTorchStreamWrapper:
        def __init__(self, pt_stream):
            self.pt_stream = pt_stream
            self.handle = pt_stream.cuda_stream

        def __cuda_stream__(self):
            stream_id = self.pt_stream.cuda_stream
            return (0, stream_id)  # Return format required by CUDA Python

    def torchrun_uid_init():
        """
        Initialize NVSHMEM using UniqueID with `torchrun` as the launcher
        """
        # Set Torch device
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        # nvshmem4py requires a cuda.core Device at init time
        global dev
        dev = Device(device.index)
        dev.set_current()
        global stream
        # Get PyTorch's current stream
        pt_stream = torch.cuda.current_stream()
        stream = PyTorchStreamWrapper(pt_stream)

        # Initialize torch.distributed process group
        world_size = torch.cuda.device_count()
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=local_rank,
            world_size=world_size,
            device_id=device
        )

        # Extract rank, nranks from process group
        num_ranks = dist.get_world_size()
        rank_id = dist.get_rank()

        # Create an empty uniqueid for all ranks
        uniqueid = nvshmem.core.get_unique_id(empty=True)
        if rank_id == 0:
            # Rank 0 gets a real uniqueid
            uniqueid = nvshmem.core.get_unique_id()
            broadcast_objects = [uniqueid]
        else:
            broadcast_objects = [None]

        # We use torch.distributed.broadcast_object_list to send the UID to all ranks
        dist.broadcast_object_list(broadcast_objects, src=0)
        dist.barrier()

        nvshmem.core.init(device=dev, uid=broadcast_objects[0], rank=rank_id, nranks=num_ranks, initializer_method="uid")

    @triton.jit
    def add_kernel(x_ptr,  # *Pointer* to first input vector.
                   y_ptr,  # *Pointer* to second input vector.
                   output_ptr,  # *Pointer* to output vector.
                   n_elements,  # Size of the vector.
                   BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                   # NOTE: `constexpr` so it can be used as a shape value.
                   ):
        """
        Addition kernel borrowed from https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
        """
        # There are multiple 'programs' processing different data. We identify which program
        # we are here:
        pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
        # This program will process inputs that are offset from the initial data.
        # For instance, if you had a vector of length 256 and block_size of 64, the programs
        # would each access the elements [0:64, 64:128, 128:192, 192:256].
        # Note that offsets is a list of pointers:
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # Create a mask to guard memory operations against out-of-bounds accesses.
        mask = offsets < n_elements
        # Load x and y from DRAM, masking out any extra elements in case the input is not a
        # multiple of the block size.
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        # Write x + y back to DRAM.
        tl.store(output_ptr + offsets, output, mask=mask)

    if __name__ == '__main__':
        torchrun_uid_init()

        """
        Allocate 3 tensors on the NVSHMEM symmetric heap
        We will add tensor1 to tensor2, and store that to tensor_out
        Then, we will use nvshmem.core to sum-reduce all PEs' copies of tensor_out
        """
        n_elements = 867530
        tensor1 = nvshmem.core.tensor((n_elements,), dtype=torch.float32)
        tensor1[:] = nvshmem.core.my_pe() + 1
        tensor2 = nvshmem.core.tensor((n_elements,), dtype=torch.float32)
        tensor2[:] = nvshmem.core.my_pe() + 2
        tensor_out = nvshmem.core.tensor((n_elements,), dtype=torch.float32)

        """
        Launch the vector addition kernel
        """
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        # This gets launched on the Torch current stream
        add_kernel[grid](tensor1, tensor2, tensor_out, n_elements, BLOCK_SIZE=1024)
        # If you uncomment this, you need to add torch.cuda.synchronize() first
        # print(f"From {nvshmem.core.my_pe()} intermediate output: {tensor_out}")

        """
        use nvshmem.core to reduce (sum) all the copies of tensor_out
        No need to synchronize, because both operations are on the same Stream
        """
        nvshmem.core.reduce(nvshmem.core.Teams.TEAM_WORLD, tensor_out, tensor_out, "sum", stream=stream)
        if nvshmem.core.my_pe() == 0:
            expected_val = 0
            for i in range(nvshmem.core.n_pes()):
                expected_val += (i + 1)
                expected_val += (i + 2)

            expected_tensor = torch.zeros_like(tensor_out)
            expected_tensor[:] = expected_val
            torch.cuda.synchronize()
            torch.testing.assert_close(tensor_out, expected_tensor)
            print(f"Final output: {tensor_out}")

        nvshmem.core.free_tensor(tensor1)
        nvshmem.core.free_tensor(tensor2)
        nvshmem.core.free_tensor(tensor_out)
        nvshmem.core.finalize()
        dist.destroy_process_group()
