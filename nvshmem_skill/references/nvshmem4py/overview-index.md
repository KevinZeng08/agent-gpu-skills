# Python Bindings (NVSHMEM4Py)

  * NVSHMEM4Py Overview
    * Current API Support
    * Key Features
    * Usage Model
    * Limitations
    * Compatibility Guide
  * Initialization and Finalization
    * Examples
    * API Reference
    * Teams
    * Initialization Methods
    * Querying Initialization Status
    * Finalization
    * Retrieving Version Information
    * Retrieving a Unique ID
  * Memory Management
    * Symmetric Memory in Python
    * Memory Lifecycle
    * Memory API reference
  * Interoperability
    * Interoperability with PyTorch and CuPy
      * Creating NVSHMEM-backed Arrays
      * NVSHMEM Operations with PyTorch and CuPy
      * Memory Management
    * Interoperability API reference
      * Interoperability with PyTorch
      * Interoperability with CuPy
  * Collective Operations
    * Supported Collective Operations
      * Reduce:
      * Broadcast:
      * ReduceScatter:
      * FCollect:
      * AlltoAll:
    * Stream Requirement for Collectives
    * Collective Examples
    * API Reference
  * Remote Memory Access (RMA)
    * RMA Operations in NVSHMEM4Py
    * Supported RMA Operations
    * Stream Requirement for RMA
    * Memory Management for RMA
    * RMA Examples
    * RMA API reference
  * Utility Functions for NVSHMEM4Py
  * Python Device APIs
    * Python Device APIs for Numba-CUDA DSL
      * NVSHMEM Device Collectives with Numba-CUDA DSL
        * Example: Using barrier and broadcast in a Numba-CUDA kernel
      * NVSHMEM Device Remote Memory Access (RMA) with Numba-CUDA DSL
        * Example: Using put and get in a Numba-CUDA kernel
      * NVSHMEM4Py Memory Management with Numba-CUDA DSL
      * NVSHMEM Device Atomic Memory Operations with Numba-CUDA DSL
        * Example: Using atomic_add in a Numba-CUDA kernel
      * Available APIs
    * Python Device APIs for CuTe DSL
      * NVSHMEM Device Collectives with CuTe DSL
        * Example: Using `barrier_all` and `broadcast` in a CuTe kernel
      * NVSHMEM Device Remote Memory Access (RMA) with CuTe DSL
        * Example: Using `put` and `get` in a CuTe kernel
      * NVSHMEM4Py Memory Management with CuTe DSL
      * NVSHMEM Device Atomic Memory Operations with CuTe DSL
        * Example: Using `atomic_add` in a CuTe kernel
      * Available APIs

NVSHMEM4Py is the official Python language binding for NVSHMEM, providing a Pythonic interface to the NVSHMEM library’s functionality. It enables Python applications to leverage NVSHMEM’s high-performance PGAS (Partitioned Global Address Space) programming model for GPU-accelerated computing.

## Quick Start

To use NVSHMEM4Py in your Python application:

    import nvshmem.core as nvshmem
    from mpi4py import MPI
    from cuda.core.experimental import Device

    dev = Device()
    dev.set_current()
    stream = dev.create_stream()

    # Initialize MPI
    comm = MPI.COMM_WORLD

    # Initialize NVSHMEM with MPI
    nvshmem.init(dev, mpi_comm=comm, init_method="mpi")

    # Get information about the current PE
    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    # Allocate symmetric memory
    # array() returns a CuPy NDArray object
    x = nvshmem.array((1024,), dtype="float32")
    y = nvshmem.array((1024,), dtype="float32")

    if my_pe == 0:
        y[:] = 1.0

    # Perform communication operations
    # Put y from PE 0 into x on PE 1
    if my_pe == 0:
        nvshmem.put(x, y, pe=1, stream=stream)

    # Synchronize PEs
    stream.sync()

    # Clean up
    nvshmem.free_array(x)
    nvshmem.free_array(y)
    nvshmem.finalize()

## Key Features

  * Pythonic interface to NVSHMEM functionality
  * Seamless integration with NumPy, CuPy, and PyTorch
  * Support for symmetric memory allocation and management
  * Communication operations (put/get, collectives)
  * Synchronization primitives

For more detailed information, see the NVSHMEM4Py Overview and API reference sections.
