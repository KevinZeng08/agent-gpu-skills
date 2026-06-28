# NVSHMEM 3.7.0 Reference Index

Curated navigation for the NVSHMEM 3.7.0 documentation mirror. For the full
C/C++ API symbol inventory (every routine, by section), see
[`api/index.md`](api/index.md).

## How to search

Search the Markdown, then open the matching page. Examples:

```bash
rg -n "nvshmem_put_signal" .              # a routine
rg -n -A4 "^`NVSHMEM_SYMMETRIC_SIZE`" env.md   # an env var + type/default
rg -rl "symmetric heap|active set" .      # pages covering a topic
rg -n "nvshmem.core|barrier_all" nvshmem4py/   # a Python binding
```

## Topic map

### Getting started / concepts
- [`introduction.md`](introduction.md) - what NVSHMEM is, key features, communication transports, GPU-initiated communication and strong scaling.
- [`using.md`](using.md) - example program, GPUDirect Async (IBGDA), using NVSHMEM with MPI/OpenSHMEM, compiling and running, communication model, data consistency, multiprocess GPU support.
- [`cuda-model.md`](cuda-model.md) - NVSHMEM and the CUDA execution model: work submission, nonlocal operations, circular dependencies, intra-kernel sync, cooperative kernel launch, implicitly async `cudaMemcpy`.
- [`tma.md`](tma.md) - using Tensor Memory Accelerator (TMA) with NVSHMEM: motivation, enabling, limitations.
- [`memory-model.md`](memory-model.md) - pointers to symmetric objects, ordering of operations, atomicity guarantees, differences from OpenSHMEM.
- [`execution-model.md`](execution-model.md) - progress of NVSHMEM operations, invoking NVSHMEM operations.
- [`constants.md`](constants.md) - library constants (versions, thread levels, comparison/signal operators, ...).
- [`handles.md`](handles.md) - predefined team handles (`NVSHMEM_TEAM_WORLD`, `NVSHMEM_TEAM_SHARED`, `NVSHMEMX_TEAM_NODE`).

### Environment variables - [`env.md`](env.md)
Grouped into Standard, Bootstrap, Additional, Collectives, Transport, and NVTX
options. Each entry lists Type, Default, and description. Jump to a variable
with `rg -n "^`NVSHMEM_..." env.md`.

### C/C++ API reference - [`api/`](api/)
- [`api/index.md`](api/index.md) - full API table of contents (every routine, by section).
- [`api/overview.md`](api/overview.md) - supported/unsupported OpenSHMEM APIs, CPU- vs GPU-thread extensions, tile-granular collectives.
- [`api/setup.md`](api/setup.md) - library setup/exit/query (`nvshmem_init`, `nvshmemx_init_attr`, unique-id / MPI / OpenSHMEM bootstrap, `nvshmem_my_pe`/`n_pes`, `nvshmem_ptr`, `nvshmemx_mc_ptr`, thread support, version/name queries).
- [`api/launch.md`](api/launch.md) - collective kernel launch (`nvshmemx_collective_launch`, `..._query_gridsize`).
- [`api/memory.md`](api/memory.md) - symmetric memory (`nvshmem_malloc`/`free`/`align`/`calloc`) and buffer registration (`nvshmemx_buffer_register*`).
- [`api/qp.md`](api/qp.md) - Queue-Pair (QP) device APIs: `nvshmemx_qp_create`, QP RMA (`qp_put`/`get`/`p`/`g`/`_nbi`), signaling (`qp_put_signal`, `qp_signal_op`), ordering (`qp_fence`, `qp_quiet`).
- [`api/teams.md`](api/teams.md) - team management: `nvshmem_team_my_pe`/`n_pes`, `nvshmem_team_split_strided`/`split_2d`, `nvshmem_team_translate_pe`, `nvshmem_team_destroy`, `nvshmemx_team_init`.
- [`api/rma.md`](api/rma.md) - remote memory access: blocking (`nvshmem_put`/`p`/`iput`, `get`/`g`/`iget`, `putmem`/`putSIZE`, `_on_stream`/`_block`/`_warp`), nonblocking (`put_nbi`/`get_nbi`), tile RMA (`tile_put`/`tile_get`).
- [`api/amo.md`](api/amo.md) - atomic memory operations: fetching (`atomic_fetch`, `fetch_inc/add/and/or/xor`, `compare_swap`, `swap`) and non-fetching (`atomic_set`, `inc/add/and/or/xor`).
- [`api/signal.md`](api/signal.md) - signaling operations: `nvshmem_put_signal`/`_nbi`, `nvshmem_signal_fetch`, `nvshmemx_signal`/`signal_op`, signal operators.
- [`api/collectives.md`](api/collectives.md) - collectives: `nvshmem_barrier`/`barrier_all`, `sync`/`sync_all`, `broadcast`, `fcollect`, `alltoall`, reductions (`{sum,prod,min,max,and,or,xor}_reduce`), and tile collectives (`tile_sum_reduce`, `tile_allgather`, `tile_broadcast`, `tile_wait`).
- [`api/sync.md`](api/sync.md) - point-to-point synchronization: `nvshmem_wait_until` (+ `_all`/`_any`/`_some`/`_vector`), `nvshmem_test` (+ variants), `nvshmem_signal_wait_until`, comparison operators.
- [`api/ordering.md`](api/ordering.md) - memory ordering: `nvshmem_fence`, `nvshmem_quiet`, `nvshmemx_flush`.

### Python bindings (NVSHMEM4Py) - [`nvshmem4py/`](nvshmem4py/)
- [`nvshmem4py/index.md`](nvshmem4py/index.md) / [`nvshmem4py/overview.md`](nvshmem4py/overview.md) - overview, current API support, usage model, limitations, compatibility guide.
- [`nvshmem4py/initialization.md`](nvshmem4py/initialization.md) - teams, init methods, status query, finalization, version, unique-id.
- [`nvshmem4py/memory_management.md`](nvshmem4py/memory_management.md) - symmetric memory in Python, memory lifecycle (`nvshmem.array`, `nvshmem.free`).
- [`nvshmem4py/interoperability.md`](nvshmem4py/interoperability.md) - PyTorch and CuPy interop, NVSHMEM-backed arrays.
- [`nvshmem4py/collectives.md`](nvshmem4py/collectives.md) - host-initiated collectives (`reduce`, `broadcast`, `reducescatter`, `fcollect`, `alltoall`) with stream requirement.
- [`nvshmem4py/rma.md`](nvshmem4py/rma.md) - host-initiated RMA operations and memory management.
- [`nvshmem4py/utils.md`](nvshmem4py/utils.md) - utility functions.
- [`nvshmem4py/device/`](nvshmem4py/device/) - device-side APIs for the **Numba-CUDA** ([`device/numba/`](nvshmem4py/device/numba/)) and **CuTe** ([`device/cute/`](nvshmem4py/device/cute/)) DSLs: collectives, RMA, atomics, memory management.

### Examples - [`examples.md`](examples.md)
End-to-end C/CUDA examples: attribute-based init, collective launch, on-stream,
threadgroup, put-on-block, TMA put, threadgroup fence+signal, ring broadcast,
ring allreduce, user-buffer registration, GEMM + AllReduce fused kernel.
Python (NVSHMEM4Py) examples: [`examples/python.md`](examples/python.md).

### Troubleshooting / FAQs - [`faq.md`](faq.md)
General, prerequisite, running, MPI/OpenSHMEM interop, GPU-GPU interconnect,
API usage, debugging, and miscellaneous FAQs.
