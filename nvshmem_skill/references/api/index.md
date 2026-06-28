# NVSHMEM APIs

  * Overview of the APIs
    * Unsupported OpenSHMEM 1.3 APIs
    * OpenSHMEM 1.3 APIs Not Supported Over Remote Network Transports
    * Supported OpenSHMEM APIs (OpenSHMEM 1.4 and 1.5)
    * NVSHMEM API Extensions For CPU Threads
    * NVSHMEM API Extensions For GPU Threads
    * Tile-Granular Collective APIs
  * Library Setup, Exit, and Query
    * **NVSHMEM_INIT**
    * **NVSHMEMX_INIT_ATTR**
    * **NVSHMEMX_HOSTLIB_INIT_ATTR**
    * **NVSHMEMX_HOSTLIB_FINALIZE**
    * **NVSHMEMX_GET_UNIQUE_ID**
    * **NVSHMEMX_SET_ATTR_UNIQUEID_ARGS**
    * **NVSHMEMX_CUMODULE_INIT**
    * **NVSHMEMX_INIT_STATUS**
    * **NVSHMEM_MY_PE**
    * **NVSHMEM_N_PES**
    * **NVSHMEM_FINALIZE**
    * **NVSHMEM_GLOBAL_EXIT**
    * **NVSHMEM_PTR**
    * **NVSHMEMX_MC_PTR**
    * **NVSHMEM_INFO_GET_VERSION**
    * **NVSHMEM_INFO_GET_NAME**
    * **NVSHMEMX_VENDOR_GET_VERSION_INFO**
  * Thread Support
    * **NVSHMEM_INIT_THREAD**
    * **NVSHMEM_QUERY_THREAD**
  * Kernel Launch Routines
    * **NVSHMEMX_COLLECTIVE_LAUNCH**
    * **NVSHMEMX_COLLECTIVE_LAUNCH_QUERY_GRIDSIZE**
  * Memory Management
    * **NVSHMEM_MALLOC, NVSHMEM_FREE, NVSHMEM_ALIGN**
    * **NVSHMEM_CALLOC**
    * Memory Registration
      * **NVSHMEMX_BUFFER_REGISTER**
      * **NVSHMEMX_BUFFER_UNREGISTER**
      * **NVSHMEMX_BUFFER_UNREGISTER_ALL**
      * **NVSHMEMX_BUFFER_REGISTER_SYMMETRIC**
      * **NVSHMEMX_BUFFER_REGISTER_SYMMETRIC_AT_PREFERRED_ADDRESS**
      * **NVSHMEMX_BUFFER_UNREGISTER_SYMMETRIC**
  * Queue Pair (QP) Specific APIs
    * Overview
    * QP Handle
    * Special QP Handle Values
    * Special PE Values for Synchronization
    * Special QP Values for Synchronization
    * API Categories
    * Device-Side Only
    * Ordering Semantics
    * Thread Safety
    * Transport Support and Compatibility
      * **NVSHMEMX_QP_CREATE**
    * QP Remote Memory Access
      * **NVSHMEMX_QP_PUT**
      * **NVSHMEMX_QP_P**
      * **NVSHMEMX_QP_GET**
      * **NVSHMEMX_QP_G**
      * **NVSHMEMX_QP_PUT_NBI**
      * **NVSHMEMX_QP_GET_NBI**
    * QP Signaling Operations
      * **NVSHMEMX_QP_SIGNAL_OP**
      * **NVSHMEMX_QP_PUT_SIGNAL**
      * **NVSHMEMX_QP_PUT_SIGNAL_NBI**
    * QP Memory Ordering
      * **NVSHMEMX_QP_FENCE**
      * **NVSHMEMX_QP_QUIET**
  * Team Management
    * Predefined and Application-Defined Teams
    * Team Handles
    * Thread Safety
    * Collective Ordering
    * Team Creation
      * Team Splitting
      * Arbitrary Team Initialization
    * **NVSHMEM_TEAM_MY_PE**
    * **NVSHMEM_TEAM_N_PES**
    * **NVSHMEM_TEAM_CONFIG_T**
    * **NVSHMEM_TEAM_GET_CONFIG**
    * **NVSHMEM_TEAM_TRANSLATE_PE**
    * **NVSHMEM_TEAM_SPLIT_STRIDED**
    * **NVSHMEM_TEAM_SPLIT_2D**
    * **NVSHMEM_TEAM_DESTROY**
    * **NVSHMEMX_TEAM_INIT**
    * **NVSHMEMX_TEAM_GET_UNIQUEID**
  * Remote Memory Access
    * Blocking RMA
      * **NVSHMEM_PUT**
      * **NVSHMEM_P**
      * **NVSHMEM_IPUT**
      * **NVSHMEM_GET**
      * **NVSHMEM_G**
      * **NVSHMEM_IGET**
    * Nonblocking RMA
      * **NVSHMEM_PUT_NBI**
      * **NVSHMEM_GET_NBI**
    * Tile RMA
      * **TILE_PUT**
      * **TILE_GET**
  * Atomic Memory Operations
    * **NVSHMEM_ATOMIC_FETCH**
    * **NVSHMEM_ATOMIC_SET**
    * **NVSHMEM_ATOMIC_COMPARE_SWAP**
    * **NVSHMEM_ATOMIC_SWAP**
    * **NVSHMEM_ATOMIC_FETCH_INC**
    * **NVSHMEM_ATOMIC_INC**
    * **NVSHMEM_ATOMIC_FETCH_ADD**
    * **NVSHMEM_ATOMIC_ADD**
    * **NVSHMEM_ATOMIC_FETCH_AND**
    * **NVSHMEM_ATOMIC_AND**
    * **NVSHMEM_ATOMIC_FETCH_OR**
    * **NVSHMEM_ATOMIC_OR**
    * **NVSHMEM_ATOMIC_FETCH_XOR**
    * **NVSHMEM_ATOMIC_XOR**
  * Signaling Operations
    * Atomicity Guarantees for Signaling Operations
    * Available Signal Operators
    * **NVSHMEM_PUT_SIGNAL**
    * **NVSHMEM_PUT_SIGNAL_NBI**
    * **NVSHMEM_SIGNAL_FETCH**
    * **NVSHMEMX_SIGNAL**
    * **NVSHMEMX_SIGNAL_OP**
  * Collective Communication
    * Team-based collectives
    * Implicit team collectives
    * Tile-based Collectives
      * Tile helper functions
      * Tile collective algorithms
    * Error codes returned from team-based collectives
    * Collective operations scopes and active sets
    * **NVSHMEM_BARRIER**
    * **NVSHMEM_SYNC**
    * **NVSHMEM_SYNC_ALL**
    * **NVSHMEM_ALLTOALL**
    * **NVSHMEM_BROADCAST**
    * **NVSHMEM_FCOLLECT**
    * **NVSHMEM_REDUCTIONS**
      * AND
      * OR
      * XOR
      * MAX
      * MIN
      * SUM
      * PROD
    * **TILE_REDUCTIONS**
    * **TILE_ALLGATHER**
    * **TILE_BROADCAST**
    * **TILE_WAIT**
  * Point-To-Point Synchronization
    * **NVSHMEM_WAIT_UNTIL**
    * **NVSHMEM_WAIT_UNTIL_ALL**
    * **NVSHMEM_WAIT_UNTIL_ANY**
    * **NVSHMEM_WAIT_UNTIL_SOME**
    * **NVSHMEM_WAIT_UNTIL_ALL_VECTOR**
    * **NVSHMEM_WAIT_UNTIL_ANY_VECTOR**
    * **NVSHMEM_WAIT_UNTIL_SOME_VECTOR**
    * **NVSHMEM_TEST**
    * **NVSHMEM_TEST_ALL**
    * **NVSHMEM_TEST_ANY**
    * **NVSHMEM_TEST_SOME**
    * **NVSHMEM_TEST_ALL_VECTOR**
    * **NVSHMEM_TEST_ANY_VECTOR**
    * **NVSHMEM_TEST_SOME_VECTOR**
    * **NVSHMEM_SIGNAL_WAIT_UNTIL**
  * Memory Ordering
    * **NVSHMEM_FENCE**
    * **NVSHMEM_QUIET**
    * **NVSHMEMX_FLUSH**
  * Language Bindings
    * Python Bindings (NVSHMEM4Py)
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
      * Quick Start
      * Key Features
