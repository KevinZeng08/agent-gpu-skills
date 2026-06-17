# NCCL 2.30 Reference Index

Curated navigation for the NCCL 2.30.7 user-guide mirror. For the exhaustive,
auto-generated deep table of contents (every heading, with anchors), see
[`index.md`](index.md).

## How to search

Search the Markdown, then open the matching page. Examples:

```bash
rg -n "ncclCommSplit" .                 # a C function
rg -n -A6 "^### NCCL_ALGO" env.md       # an env var + accepted values
rg -rl "fault tolerance" .              # pages covering a topic
rg -n "Communicator.allreduce" nccl4py/ # a Python binding method
```

## Topic map

### Getting started
- [`overview.md`](overview.md) - what NCCL is and what it provides.
- [`setup.md`](setup.md) - installation and linking.
- [`examples.md`](examples.md) - end-to-end communicator and communication examples.
- [`nccl1.md`](nccl1.md) - migrating from NCCL 1 to NCCL 2.
- [`mpi.md`](mpi.md) - using NCCL together with MPI.

### Using NCCL (concepts) - [`usage/`](usage/)
- [`usage/communicators.md`](usage/communicators.md) - creating / splitting / shrinking / growing / destroying communicators; error handling and communicator abort; fault tolerance; Quality of Service.
- [`usage/collectives.md`](usage/collectives.md) - AllReduce, Broadcast, Reduce, AllGather, ReduceScatter, AlltoAll, Gather, Scatter (with diagrams).
- [`usage/data.md`](usage/data.md) - data pointers.
- [`usage/streams.md`](usage/streams.md) - CUDA stream semantics.
- [`usage/groups.md`](usage/groups.md) - group calls, aggregation, ordering, nonblocking groups.
- [`usage/p2p.md`](usage/p2p.md) - point-to-point: two-sided (sendrecv, scatter, gather, all-to-all, neighbor exchange) and one-sided (put/signal/wait, barrier).
- [`usage/threadsafety.md`](usage/threadsafety.md) - thread safety.
- [`usage/inplace.md`](usage/inplace.md) - in-place operations.
- [`usage/cudagraph.md`](usage/cudagraph.md) - using NCCL with CUDA graphs.
- [`usage/bufferreg.md`](usage/bufferreg.md) - user-buffer / window registration (NVLS, IB SHARP, PXN, memory allocator, zero-CTA).
- [`usage/deviceapi.md`](usage/deviceapi.md) - device-initiated communication (LSA, Multimem, GIN, teams, segments).

### C API reference - [`api/`](api/)
- [`api/comms.md`](api/comms.md) - communicator creation & management functions.
- [`api/colls.md`](api/colls.md) - collective communication functions.
- [`api/group.md`](api/group.md) - `ncclGroupStart` / `ncclGroupEnd` / `ncclGroupSimulateEnd`.
- [`api/p2p.md`](api/p2p.md) - two-sided and one-sided (RMA) point-to-point functions.
- [`api/types.md`](api/types.md) - `ncclComm_t`, `ncclResult_t`, `ncclDataType_t`, `ncclRedOp_t`, `ncclConfig_t`, `ncclWindow_t`, ...
- [`api/ops.md`](api/ops.md) - user-defined reduction operators (`ncclRedOpCreatePreMulSum`).
- [`api/flags.md`](api/flags.md) - window registration, CTA policy, and shrink flags.
- [`api/device.md`](api/device.md) - device API overview, linking to:
  - [`api/device_setup.md`](api/device_setup.md) - host-side setup.
  - [`api/device_memory.md`](api/device_memory.md) - device memory and LSA / Multimem.
  - [`api/device_gin.md`](api/device_gin.md) - GIN (GPU-Initiated Networking).
  - [`api/device_reducecopy.md`](api/device_reducecopy.md) - remote reduce/copy building blocks.
- [`api/param.md`](api/param.md) - NCCL parameter API (handle-based and key-based).

### Python bindings (nccl4py) - [`nccl4py/`](nccl4py/)
- [`nccl4py/communicator.md`](nccl4py/communicator.md) - `Communicator` overview, then:
  - [`nccl4py/communicator/class.md`](nccl4py/communicator/class.md) - class and properties.
  - [`nccl4py/communicator/lifecycle.md`](nccl4py/communicator/lifecycle.md) - init / split / shrink / grow / destroy.
  - [`nccl4py/communicator/collectives.md`](nccl4py/communicator/collectives.md) - collective methods.
  - [`nccl4py/communicator/p2p.md`](nccl4py/communicator/p2p.md) - send / recv / signal methods.
  - [`nccl4py/communicator/registration.md`](nccl4py/communicator/registration.md) - buffer / window registration.
  - [`nccl4py/communicator/device_setup.md`](nccl4py/communicator/device_setup.md) - device communicator setup.
  - [`nccl4py/communicator/status.md`](nccl4py/communicator/status.md) - status and utility methods.
- [`nccl4py/configuration.md`](nccl4py/configuration.md) - `NCCLConfig`, `NCCLDevCommRequirements`, `CTAPolicy`.
- [`nccl4py/group.md`](nccl4py/group.md) - `group`, `group_start`, `group_end`, `GroupSimInfo`.
- [`nccl4py/memory.md`](nccl4py/memory.md) - `mem_alloc`, `mem_free`.
- [`nccl4py/resources.md`](nccl4py/resources.md) - resource handles and custom reduction ops.
- [`nccl4py/types.md`](nccl4py/types.md) - data types, reduction operators, exceptions.
- [`nccl4py/versions.md`](nccl4py/versions.md) - version helpers.
- [`nccl4py/interop.md`](nccl4py/interop.md) - CuPy and PyTorch interop.

### Environment variables - [`env.md`](env.md)
One page, two halves: **System configuration** (keep set) and **Debugging**
(temporary). Jump to a variable with `rg -n "^### NCCL_..." env.md`.

### Troubleshooting - [`troubleshooting/`](troubleshooting/)
- [`troubleshooting/gpu_troubleshooting.md`](troubleshooting/gpu_troubleshooting.md) - GPU-to-GPU / GPU-to-NIC, ACS, topology, multi-node NVLink.
- [`troubleshooting/networking_troubleshooting.md`](troubleshooting/networking_troubleshooting.md) - interfaces, fabric checks, latency/bandwidth, InfiniBand / RoCE.
- [`troubleshooting/runtime_and_mpi_issues.md`](troubleshooting/runtime_and_mpi_issues.md) - errors, shared memory, stack size, UVM, file descriptors, MPI.
- [`troubleshooting/performance_and_tuning.md`](troubleshooting/performance_and_tuning.md) - intra/inter-node perf, MNNVL, tuning, affinity.
- [`troubleshooting/logging.md`](troubleshooting/logging.md) - logging levels, subsystem filters, files, timestamps, common scenarios.
- [`troubleshooting/ras.md`](troubleshooting/ras.md) - RAS subsystem for diagnosing hangs and crashes.
