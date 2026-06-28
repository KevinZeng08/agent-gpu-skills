# Python Device APIs for CuTe DSL

  * NVSHMEM Device Collectives with CuTe DSL
    * Example: Using `barrier_all` and `broadcast` in a CuTe kernel
  * NVSHMEM Device Remote Memory Access (RMA) with CuTe DSL
    * Example: Using `put` and `get` in a CuTe kernel
  * NVSHMEM4Py Memory Management with CuTe DSL
  * NVSHMEM Device Atomic Memory Operations with CuTe DSL
    * Example: Using `atomic_add` in a CuTe kernel

This section documents the Python device APIs for using NVSHMEM with the CuTe DSL. CuTe provides a convenient way to author GPU kernels in Python, enabling rapid prototyping and development of CUDA code. By leveraging NVSHMEM’s Python bindings, you can write CuTe kernels that perform efficient GPU-to-GPU communication, such as remote memory access (RMA) and collective operations, directly from Python.

These APIs allow you to launch kernels that use NVSHMEM primitives for communication and synchronization between GPUs, making it easier to develop scalable, high-performance applications in Python. The following pages describe the available APIs and provide usage examples for integrating NVSHMEM with your CuTe workflows.

## Available APIs

The CuTe NVSHMEM device bindings include:

  * Collectives: `barrier`, `barrier_all`, `sync`, `sync_all`, `reduce`, `reducescatter`, `fcollect`, `broadcast`, `alltoall` (plus `nvshmemx_*_{block,warp}` variants).
  * RMA: vector `put`/`get` (blocking and nonblocking `_nbi`), `put_signal`, and scalar `p`/`g`.
  * Atomics: `fetch`/`set`/`swap`/`compare_swap`, `inc`/`fetch_inc`, `add`/`fetch_add`, bitwise `and`/`fetch_and`, `or`/`fetch_or`, `xor`/`fetch_xor`.
  * Signalling: `signal_op`, `signal_wait_until`.
  * Utilities: `n_pes`, `my_pe`, `team_n_pes`, `team_my_pe`.

Execution scopes and semantics:

  * Scopes: `device`, `block`, `warp`.
  * Semantics: blocking and nonblocking (`_nbi`) where supported.

Data types:

NVSHMEM4Py CuTe APIs support passing in CuTe Tensors as arguments. Refer to CuTe documentation for more details. You can also use Torch tensors by using DLPack to convert them to CuTe Tensors.
