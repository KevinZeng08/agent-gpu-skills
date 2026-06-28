# Python Device APIs for Numba-CUDA DSL

  * NVSHMEM Device Collectives with Numba-CUDA DSL
    * Example: Using barrier and broadcast in a Numba-CUDA kernel
  * NVSHMEM Device Remote Memory Access (RMA) with Numba-CUDA DSL
    * Example: Using put and get in a Numba-CUDA kernel
  * NVSHMEM4Py Memory Management with Numba-CUDA DSL
  * NVSHMEM Device Atomic Memory Operations with Numba-CUDA DSL
    * Example: Using atomic_add in a Numba-CUDA kernel

This section documents the Python device APIs for using NVSHMEM with the Numba-CUDA DSL. Numba provides a convenient way to author GPU kernels in Python, enabling rapid prototyping and development of CUDA code. By leveraging NVSHMEM’s Python bindings, you can write Numba-CUDA kernels that perform efficient GPU-to-GPU communication, such as remote memory access (RMA) and collective operations, directly from Python.

These APIs allow you to launch kernels that use NVSHMEM primitives for communication and synchronization between GPUs, making it easier to develop scalable, high-performance applications in Python. The following pages describe the available APIs and provide usage examples for integrating NVSHMEM with your Numba-CUDA workflows.

## Available APIs

The Numbast-generated NVSHMEM device bindings include:

  * Collectives: `barrier`, `barrier_all`, `sync`, `sync_all`, `reduce`, `reducescatter`, `fcollect`, `broadcast`, `alltoall` (plus `nvshmemx_*_{block,warp}` variants).
  * RMA: vector `put`/`get` (blocking and nonblocking `_nbi`), `put_signal`, and scalar `p`/`g`.
  * Atomics: `fetch`/`set`/`swap`/`compare_swap`, `inc`/`fetch_inc`, `add`/`fetch_add`, bitwise `and`/`fetch_and`, `or`/`fetch_or`, `xor`/`fetch_xor`.
  * Signalling: `signal_op`, `signal_wait_until`.
  * Utilities: `n_pes`, `my_pe`, `team_n_pes`, `team_my_pe`.

Execution scopes and semantics:

  * Scopes: `device`, `block`, `warp`.
  * Semantics: blocking and nonblocking (`_nbi`) where supported.

Data types:

NVSHMEM4Py Numba APIs support passing in CuPy arrays as arguments. Refer to Cupy documentation and Numba-Cuda documentation for more details.
