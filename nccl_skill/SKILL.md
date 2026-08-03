---
name: nccl-skill
description: >-
  NVIDIA NCCL 2.30 reference: collective and point-to-point GPU communication
  (AllReduce, Broadcast, Reduce, AllGather, ReduceScatter, AlltoAll, Gather,
  Scatter), communicator lifecycle (ncclCommInitRank, ncclCommSplit,
  ncclCommShrink, ncclCommGrow, ncclCommAbort), the C API, the nccl4py Python
  bindings, device-initiated communication (LSA, Multimem, GIN), CUDA graphs,
  user-buffer/window registration, environment variables (NCCL_DEBUG,
  NCCL_SOCKET_IFNAME, NCCL_IB_HCA, NCCL_ALGO, NCCL_PROTO, ...), RAS, and
  troubleshooting hangs/perf over NVLink, PCIe, InfiniBand and RoCE. Use when
  working with NCCL or nccl4py, tuning NCCL_* variables, debugging multi-GPU /
  multi-node hangs or slow collectives, or looking up any ncclXxx function,
  type, or flag.
---

# NCCL 2.30 Documentation

A searchable, offline mirror of the NVIDIA Collective Communication Library
(NCCL) **2.30.7** user guide. Source:
<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/>

## How to use this skill

The full guide is split into grep-able Markdown pages under
[`references/`](references/). Do not read everything; **search** for the
symbol or topic, then open the matching page.

```bash
# from this skill directory
rg -n "ncclCommSplit" references/            # find a function
rg -n "NCCL_IB_HCA" references/env.md        # find an env var
rg -rl "fault tolerance" references/          # find pages on a topic
```

- Full deep table of contents: [`references/index.md`](references/index.md)
- Curated topic map + more search recipes: [`references/REFERENCE_INDEX.md`](references/REFERENCE_INDEX.md)

## Reference map

| Area | Page(s) |
|------|---------|
| Overview, setup | [`references/overview.md`](references/overview.md), [`references/setup.md`](references/setup.md) |
| Using NCCL (concepts) | [`references/usage/`](references/usage/) |
| C API reference | [`references/api/`](references/api/) |
| Python bindings (nccl4py) | [`references/nccl4py/`](references/nccl4py/) |
| Environment variables | [`references/env.md`](references/env.md) |
| Troubleshooting / RAS | [`references/troubleshooting/`](references/troubleshooting/) |
| Migrating NCCL 1 -> 2 | [`references/nccl1.md`](references/nccl1.md) |
| NCCL + MPI | [`references/mpi.md`](references/mpi.md) |
| Worked examples | [`references/examples.md`](references/examples.md) |

## Quick reference

### Collective operations

| Operation | C function | nccl4py method | Concept page |
|-----------|------------|----------------|--------------|
| AllReduce | `ncclAllReduce` | `Communicator.allreduce` | `usage/collectives.md#allreduce` |
| Broadcast | `ncclBroadcast` | `Communicator.broadcast` | `usage/collectives.md#broadcast` |
| Reduce | `ncclReduce` | `Communicator.reduce` | `usage/collectives.md#reduce` |
| AllGather | `ncclAllGather` | `Communicator.allgather` | `usage/collectives.md#allgather` |
| ReduceScatter | `ncclReduceScatter` | `Communicator.reduce_scatter` | `usage/collectives.md#reducescatter` |
| AlltoAll | `ncclAlltoAll` | `Communicator.alltoall` | `usage/collectives.md#alltoall` |
| Gather | `ncclGather` | `Communicator.gather` | `usage/collectives.md#gather` |
| Scatter | `ncclScatter` | `Communicator.scatter` | `usage/collectives.md#scatter` |

C signatures: [`references/api/colls.md`](references/api/colls.md) ·
nccl4py: [`references/nccl4py/communicator/collectives.md`](references/nccl4py/communicator/collectives.md)

### Communicator lifecycle (C API)

`ncclGetUniqueId` -> `ncclCommInitRank` / `ncclCommInitRankConfig` /
`ncclCommInitAll` / `ncclCommInitRankScalable` -> ... ->
`ncclCommFinalize` -> `ncclCommDestroy`.

Reconfigure / fault-handling: `ncclCommSplit`, `ncclCommShrink`,
`ncclCommGrow`, `ncclCommRevoke`, `ncclCommAbort`, `ncclCommGetAsyncError`,
`ncclCommSuspend` / `ncclCommResume`.
Full signatures: [`references/api/comms.md`](references/api/comms.md).
Concepts (including error handling, fault tolerance, QoS):
[`references/usage/communicators.md`](references/usage/communicators.md).

### Point-to-point communication

- Two-sided: `ncclSend`, `ncclRecv` (wrap in `ncclGroupStart` /
  `ncclGroupEnd` for sendrecv / scatter / gather / all-to-all).
- One-sided (RMA): `ncclPutSignal`, `ncclSignal`, `ncclWaitSignal`.

C: [`references/api/p2p.md`](references/api/p2p.md) ·
Concepts: [`references/usage/p2p.md`](references/usage/p2p.md)

### Data types and reduction operators

- `ncclRedOp_t`: `ncclSum`, `ncclProd`, `ncclMin`, `ncclMax`, `ncclAvg`
  (custom pre-multiply via `ncclRedOpCreatePreMulSum` /
  `ncclRedOpDestroy`, see [`references/api/ops.md`](references/api/ops.md)).
- `ncclDataType_t`: `ncclInt8`/`ncclChar`, `ncclUint8`, `ncclInt32`/`ncclInt`,
  `ncclUint32`, `ncclInt64`, `ncclUint64`, `ncclFloat16`/`ncclHalf`,
  `ncclFloat32`/`ncclFloat`, `ncclFloat64`/`ncclDouble`, `ncclBfloat16`,
  `ncclFloat8e4m3`, `ncclFloat8e5m2`.

All types and flags: [`references/api/types.md`](references/api/types.md),
[`references/api/flags.md`](references/api/flags.md).

### Group calls, streams, CUDA graphs

`ncclGroupStart` / `ncclGroupEnd` (and `ncclGroupSimulateEnd`) aggregate or
fuse operations: [`references/api/group.md`](references/api/group.md),
[`references/usage/groups.md`](references/usage/groups.md). Stream semantics:
[`references/usage/streams.md`](references/usage/streams.md). CUDA graph
capture: [`references/usage/cudagraph.md`](references/usage/cudagraph.md).

### Device-initiated communication

GPU-side communication from within kernels: LSA (load/store accessible),
Multimem (NVLink SHARP), and GIN (GPU-Initiated Networking), plus
remote reduce/copy building blocks for fused kernels.

- Concepts: [`references/usage/deviceapi.md`](references/usage/deviceapi.md)
- Host-side setup: [`references/api/device_setup.md`](references/api/device_setup.md)
- Memory & LSA / Multimem: [`references/api/device_memory.md`](references/api/device_memory.md)
- GIN: [`references/api/device_gin.md`](references/api/device_gin.md)
- Reduce/Copy building blocks: [`references/api/device_reducecopy.md`](references/api/device_reducecopy.md)

### User-buffer / window registration

`ncclCommRegister` / `ncclCommDeregister`,
`ncclCommWindowRegister` / `ncclCommWindowDeregister`,
`ncclMemAlloc` / `ncclMemFree`. Concepts (NVLS, IB SHARP, PXN, zero-CTA):
[`references/usage/bufferreg.md`](references/usage/bufferreg.md).

### Environment variables (by category)

Full list with accepted values: [`references/env.md`](references/env.md).

| Category | Common variables |
|----------|------------------|
| Network / system | `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `NCCL_NET`, `NCCL_NET_PLUGIN`, `NCCL_CROSS_NIC` |
| Logging / debug | `NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS`, `NCCL_DEBUG_FILE` |
| Algorithm / perf | `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_NTHREADS`, `NCCL_MIN_NCHANNELS`, `NCCL_MAX_NCHANNELS`, `NCCL_BUFFSIZE` |
| Transports / GDR | `NCCL_P2P_LEVEL`, `NCCL_P2P_DISABLE`, `NCCL_SHM_DISABLE`, `NCCL_IB_DISABLE`, `NCCL_NET_GDR_LEVEL` |
| Features | `NCCL_NVLS_ENABLE`, `NCCL_CUMEM_ENABLE`, `NCCL_MNNVL_ENABLE`, `NCCL_RAS_ENABLE` |
| Behavior | `NCCL_COMM_BLOCKING`, `NCCL_LAUNCH_MODE`, `NCCL_CTA_POLICY` |

### Python bindings (nccl4py)

`Communicator` class and lifecycle/collective/p2p/registration methods,
`NCCLConfig`, group context managers, `mem_alloc`/`mem_free`, CuPy & PyTorch
interop. Entry point: [`references/nccl4py/communicator.md`](references/nccl4py/communicator.md);
config: [`references/nccl4py/configuration.md`](references/nccl4py/configuration.md);
interop: [`references/nccl4py/interop.md`](references/nccl4py/interop.md).

## Troubleshooting quick links

| Symptom | Start here |
|---------|------------|
| Hang at init / unknown stall | [`references/troubleshooting/ras.md`](references/troubleshooting/ras.md), [`references/troubleshooting/runtime_and_mpi_issues.md`](references/troubleshooting/runtime_and_mpi_issues.md) |
| Wrong/missing NIC, IB/RoCE, GID | [`references/troubleshooting/networking_troubleshooting.md`](references/troubleshooting/networking_troubleshooting.md) |
| GPU Direct, ACS, topology | [`references/troubleshooting/gpu_troubleshooting.md`](references/troubleshooting/gpu_troubleshooting.md) |
| Slow collectives / tuning | [`references/troubleshooting/performance_and_tuning.md`](references/troubleshooting/performance_and_tuning.md) |
| What is NCCL logging telling me | [`references/troubleshooting/logging.md`](references/troubleshooting/logging.md) |

## Search recipes

```bash
# A function's full signature and description
rg -n -A8 "ncclCommInitRankConfig" references/api/comms.md

# Every page that mentions a concept
rg -rl "user buffer registration|window registration" references/

# An environment variable and its accepted values
rg -n -A6 "^### NCCL_ALGO" references/env.md

# A nccl4py method
rg -n -A10 "Communicator.reduce_scatter" references/nccl4py/

# Device API building blocks
rg -n "ReduceSumCopy|Multimem|GIN|LSA" references/api/ references/usage/deviceapi.md
```

## Reading the NCCL source

The docs describe behavior; the source explains it. Run
[`update-nccl.sh`](update-nccl.sh) to fetch the matching NCCL source
(tag `v2.30.7-1`) into `repos/nccl/` for side-by-side reading:

```bash
nccl_skill/update-nccl.sh            # sparse: key dirs only (~12 MB)
nccl_skill/update-nccl.sh --full     # whole repo
```

Source map (`repos/nccl/`):

| Looking for | Source |
|-------------|--------|
| Public API surface | `src/nccl.h.in` |
| Communicator init / lifecycle | `src/init.cc`, `src/group.cc` |
| Collective entry points / launch | `src/collectives.cc`, `src/enqueue.cc` |
| Device kernels | `src/device/` |
| Device API (LSA / Multimem / GIN) | `src/include/nccl_device/`, `src/nccl_device/`, `src/gin/`, `src/rma/` |
| Topology / path search | `src/graph/` |
| Transports (P2P / SHM / NET / IB) | `src/transport/` |
| Env var parsing | `src/param/`, `src/include/param.h` |
| RAS subsystem | `src/ras/` |
| Python bindings | `bindings/nccl4py/nccl/` |

Cross-reference docs and code in one search, e.g.:

```bash
rg -n "NCCL_ALGO" references/env.md repos/nccl/src
rg -n "ncclCommShrink" references/api/comms.md repos/nccl/src
```

## Regenerating

These pages are generated from the in-repo doc source
(`docs/userguide/source`) by [`build_nccl_skill.sh`](build_nccl_skill.sh).
Re-run it after the docs change. See [`README.md`](README.md) for details.
