# Python Device APIs

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

NVSHMEM4Py brings device-side APIs to Python, letting you call NVSHMEM operations directly from GPU kernels. This works by integrating with Python device Domain-Specific Languages (DSLs) - specialized tools that let you write and run GPU kernels in Python. These DSLs give you a familiar and efficient way to handle device-level parallel programming and communication.

Right now, NVSHMEM4Py has built-in support for two device-side DSLs: Numba-CUDA and CuTe. Numba is a popular Python library that compiles Python functions on-the-fly to run on CUDA-enabled GPUs. With Numba-CUDA, you can write Python functions as CUDA kernels, and NVSHMEM4Py enables these kernels to perform NVSHMEM operations—such as remote memory access, collective operations, and synchronization—directly on the GPU. CuTe provides a DSL for expressing GPU kernels, and NVSHMEM4Py similarly integrates NVSHMEM operations within CuTe kernels for on-device communication.

This integration makes it easy for Python developers to build high-performance, distributed GPU applications using natural Python code. You get all the benefits of NVSHMEM’s one-sided communication and PGAS (Partitioned Global Address Space) model without leaving Python. As the Python GPU ecosystem grows, NVSHMEM4Py will continue broadening device DSL support, making device-side NVSHMEM programming even more accessible and flexible in Python.
