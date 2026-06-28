# Language Bindings

Contents:

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

This directory contains API specifications for official NVSHMEM language bindings.

The core NVSHMEM API is implemented in C/C++. Language bindings provide wrappers around the core C/C++ implementation to enable NVSHMEM functionality in other programming languages.
