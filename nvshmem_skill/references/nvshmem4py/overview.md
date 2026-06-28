# NVSHMEM4Py Overview

NVSHMEM4Py is the official Python language binding for NVSHMEM, providing a Pythonic interface to the NVSHMEM library. It enables Python applications to leverage the high-performance, PGAS (Partitioned Global Address Space) programming model offered by NVSHMEM for GPU-centric communication.

## Current API Support

Currently, NVSHMEM4py supports the native pythonic extensions for host-side APIs. This implies that supported NVSHMEM point-to-point/collective operations are dispatched from the host (CPU) side, potentially launching GPU communications kernels/GPU offload operations on user-defined CUDA stream. Additionally, it allows for flexibility to interop with Python domain-specific language (DSL) to author custom kernels targeting peer-to-peer communication using local and remote symmetric memory buffers. Native pythonic extensions for device-side APIs in DSLs are not yet supported.

## Key Features

  * **Seamless Integration with Python** : NVSHMEM4Py allows Python applications to utilize NVSHMEM’s capabilities with a Pythonic interface.
  * **Interoperability** : Native support for the broader CUDA Python ecosystem, with special support for PyTorch and CuPy. Additionally, NVSHMEM4Py interoperates with Python DSLs for authoring GPU Kernels, including Numba-Cuda and Triton. This enables Python developers to author GPU kernels involving communication in Python.
  * **Symmetric Memory Management** : Provides Python interfaces to allocate and manage symmetric memory across multiple GPUs.

## Usage Model

NVSHMEM4Py follows the same programming model as the core NVSHMEM library, with adaptations to make it more Pythonic. Applications typically:

  1. Initialize the NVSHMEM environment
  2. Allocate symmetric memory
  3. Perform communication operations (put/get, collectives, etc.)
  4. Synchronize as needed
  5. Finalize the NVSHMEM environment

The Python API maintains the same PE (Processing Element) concept as the core NVSHMEM library, where each PE represents a process with its associated GPU.

## Limitations

  * Not all NVSHMEM C/C++ APIs are currently exposed in the Python binding
  * Performance may have some overhead compared to the native C/C++ implementation
  * Requires proper CUDA and NVSHMEM installation on the system

NVSHMEM4Py enables Python developers to write distributed GPU applications with a simple shared memory programming model, making it easier to scale Python applications across multiple GPUs and nodes.

## Compatibility Guide

NVSHMEM4Py host and device APIs have different compatibility requirements due to dependencies of libraries they depend on.

NVSHMEM4Py host APIs depend on the following libraries:

  * cuda-core
  * cuda-bindings
  * cuda-pathfinder
  * numpy

NVSHMEM4Py device APIs additionally depend on the following libraries:

  * numba-cuda
  * cupy

Several additional libraries are supported by NVSHMEM4Py for interoperability, but are not required for the core functionality of NVSHMEM4Py:

  * cupy
  * torch
  * ml-dtypes
  * mpi4py

NVSHMEM4Py tests use pytest and pytest-mpi. If you would like to run our tests, you will need to install these libraries.

Please refer to our requirements file for more details on how to install these optional libraries.

The following table summarizes the supported version configurations for NVSHMEM4Py host and device (numba-cuda DSL) APIs.

NVSHMEM release | NVSHMEM4Py version | cuda-core | cuda-bindings | torch | numba-cuda | cuda-pathfinder | nvidia-cutlass-dsl
---|---|---|---|---|---|---|---
3.3.9 | 0.1.2 | 0.2.0 | 12.8.x/13.0.x | 2.8.0 | N/A | N/A | N/A
3.4.5 | 0.1.3 | 0.4.0 | 12.8.x/13.0.x | 2.9.0 | N/A | 1.2.3 | N/A
3.5.19 | 0.2.1 | 0.4.0 | 12.9.x/13.0.x/13.1.x | 2.9.0 | 0.20.1 | 1.2.3 | N/A
3.5.21 | 0.2.2 | 0.4.0 | 12.9.x/13.0.x/13.1.x | 2.9.0 | 0.20.1 | 1.2.3 | N/A
3.6.5 | 0.3.0 | >=0.5.0 | 12.9.x/13.1.x/13.2.x | 2.9.0 | 0.28 | >=1.2.3 | 4.4.2
3.7.0 | 0.3.1 | >=0.5.0 | 12.9.x/13.2.x/13.3.x | 2.9.0 | >=0.31.0 | >=1.2.3 | 4.4.2

Because many of these libraries follow semantic versioning’s “version zero” rule, it is recommended to use exactly the version specified in the table for the best compatibility. Other versions may work, but are not guaranteed to be supported.
