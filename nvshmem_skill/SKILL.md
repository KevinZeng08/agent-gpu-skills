---
name: nvshmem-skill
description: >-
  NVIDIA NVSHMEM 3.7.0 reference: the OpenSHMEM PGAS programming model for
  clusters of NVIDIA GPUs. Symmetric memory (nvshmem_malloc / nvshmem_calloc /
  nvshmem_align), GPU-initiated and on-stream one-sided communication — RMA
  (nvshmem_put/get, iput/iget, put_nbi/get_nbi, putmem, _block/_warp variants),
  atomics (nvshmem_atomic_add/inc/fetch_add/compare_swap/...), signaling
  (nvshmem_put_signal, signal_op, signal_wait_until), collectives (barrier,
  sync, broadcast, fcollect, alltoall, reductions sum/prod/min/max/and/or/xor),
  teams (nvshmem_team_split_strided/2d, NVSHMEM_TEAM_WORLD), point-to-point sync
  (wait_until, test), and memory ordering (fence, quiet, nvshmemx_flush). Covers
  setup/init (nvshmem_init, nvshmemx_init_attr, unique-id / MPI / OpenSHMEM
  bootstrap), collective kernel launch (nvshmemx_collective_launch), IBGDA /
  GPUDirect Async, Queue-Pair (QP) device APIs, TMA, the NVSHMEM4Py Python
  bindings (nvshmem.core, PyTorch/CuPy interop, Numba-CUDA and CuTe DSL device
  APIs), every NVSHMEM_* environment variable, and troubleshooting/FAQs. Use
  when writing or debugging NVSHMEM / OpenSHMEM GPU communication code, tuning
  NVSHMEM_* variables, multi-GPU/multi-node put/get/collective hangs, or looking
  up any nvshmem*/nvshmemx* symbol, constant, or env var.
---

# NVSHMEM 3.7.0 Documentation

A searchable, offline mirror of the NVIDIA OpenSHMEM Library (NVSHMEM)
**3.7.0** documentation. Source:
<https://docs.nvidia.com/nvshmem/api/index.html>

NVSHMEM implements the OpenSHMEM Partitioned Global Address Space (PGAS) model
across the memory of multiple NVIDIA GPUs, with fine-grained GPU-GPU data
movement issued from inside a CUDA kernel, on a CUDA stream, or from the CPU.

## How to use this skill

The docs are split into grep-able Markdown pages under
[`references/`](references/). Do not read everything; **search** for the
symbol, env var, or topic, then open the matching page.

```bash
# from this skill directory
rg -n "nvshmem_put_signal" references/            # find a function
rg -n -A4 "^`NVSHMEM_SYMMETRIC_SIZE`" references/env.md   # an env var + type/default
rg -rl "symmetric heap|symmetric memory" references/      # pages on a topic
```

- Curated topic map + more search recipes: [`references/INDEX.md`](references/INDEX.md)
- Full C/C++ API symbol inventory (which page each routine lives on): [`references/api/index.md`](references/api/index.md)

## Reference map

| Area | Page(s) |
|------|---------|
| Introduction, key features, transports | [`references/introduction.md`](references/introduction.md) |
| Using NVSHMEM (compile/run, bootstrap, IBGDA, data consistency) | [`references/using.md`](references/using.md) |
| NVSHMEM + the CUDA execution model | [`references/cuda-model.md`](references/cuda-model.md) |
| Using TMA with NVSHMEM | [`references/tma.md`](references/tma.md) |
| Memory model (symmetric objects, ordering, atomicity) | [`references/memory-model.md`](references/memory-model.md) |
| Execution model (progress, invocation) | [`references/execution-model.md`](references/execution-model.md) |
| Library constants / handles | [`references/constants.md`](references/constants.md), [`references/handles.md`](references/handles.md) |
| Environment variables (`NVSHMEM_*`) | [`references/env.md`](references/env.md) |
| C/C++ API reference | [`references/api/`](references/api/) |
| Python bindings (NVSHMEM4Py) | [`references/nvshmem4py/`](references/nvshmem4py/) |
| Worked examples (C/CUDA + Python) | [`references/examples.md`](references/examples.md), [`references/examples/python.md`](references/examples/python.md) |
| Troubleshooting / FAQs | [`references/faq.md`](references/faq.md) |

## Quick reference

### Setup, exit, query

`nvshmem_init` / `nvshmem_init_thread` -> ... -> `nvshmem_finalize`.
Attribute-based init (`nvshmemx_init_attr` with unique-id, MPI, or OpenSHMEM
bootstrap), status query (`nvshmemx_init_status`), `nvshmem_my_pe` /
`nvshmem_n_pes`, `nvshmem_ptr` / `nvshmemx_mc_ptr`, version/name queries.
See [`references/api/setup.md`](references/api/setup.md).
Collective kernel launch (`nvshmemx_collective_launch`,
`..._query_gridsize`): [`references/api/launch.md`](references/api/launch.md).

### Symmetric memory management

`nvshmem_malloc`, `nvshmem_free`, `nvshmem_align`, `nvshmem_calloc`, and buffer
registration (`nvshmemx_buffer_register` / `..._unregister` /
`..._register_symmetric`). See [`references/api/memory.md`](references/api/memory.md).

### Remote Memory Access (RMA)

| Class | Routines | Page |
|-------|----------|------|
| Blocking | `nvshmem_put` / `nvshmem_p` / `nvshmem_iput`, `nvshmem_get` / `nvshmem_g` / `nvshmem_iget` (+ `putmem`/`getmem`, `putSIZE`, `_on_stream`, `_block`, `_warp`) | [`api/rma.md`](references/api/rma.md) |
| Nonblocking | `nvshmem_put_nbi`, `nvshmem_get_nbi` | [`api/rma.md`](references/api/rma.md) |
| Tile | `tile_put`, `tile_get` (NVLink/remote, `tile_algo_t`) | [`api/rma.md`](references/api/rma.md) |

### Atomic Memory Operations (AMO)

Fetching: `nvshmem_atomic_fetch`, `..._fetch_inc/add/and/or/xor`,
`..._compare_swap`, `..._swap`. Non-fetching: `nvshmem_atomic_set`,
`..._inc/add/and/or/xor`. See [`references/api/amo.md`](references/api/amo.md).

### Signaling

`nvshmem_put_signal` / `..._nbi`, `nvshmem_signal_fetch`, `nvshmemx_signal_op`,
signal operators (`NVSHMEM_SIGNAL_SET`, `NVSHMEM_SIGNAL_ADD`). Wait on a signal:
`nvshmem_signal_wait_until`. See [`references/api/signal.md`](references/api/signal.md).

### Collective communication

`nvshmem_barrier` / `nvshmem_barrier_all`, `nvshmem_sync` / `nvshmem_sync_all`,
`nvshmem_broadcast`, `nvshmem_fcollect`, `nvshmem_alltoall`, and reductions
(`nvshmem_{sum,prod,min,max,and,or,xor}_reduce`). Tile collectives
(`tile_sum_reduce`, `tile_allgather`, `tile_broadcast`, `tile_wait`).
All are team-based with `_on_stream` / `_block` / `_warp` variants.
See [`references/api/collectives.md`](references/api/collectives.md).

### Point-to-point synchronization

`nvshmem_wait_until` (+ `_all` / `_any` / `_some` and `_vector` forms) and
`nvshmem_test` (+ `_all` / `_any` / `_some` / `_vector`). Comparison ops
`NVSHMEM_CMP_EQ/NE/GT/GE/LT/LE`. See [`references/api/sync.md`](references/api/sync.md).

### Memory ordering

`nvshmem_fence` (point-to-point ordering), `nvshmem_quiet` (completion of all
outstanding ops), `nvshmemx_flush`. See [`references/api/ordering.md`](references/api/ordering.md).

### Teams

`NVSHMEM_TEAM_WORLD`, `NVSHMEM_TEAM_SHARED`, `NVSHMEMX_TEAM_NODE`;
`nvshmem_team_my_pe` / `..._n_pes`, `nvshmem_team_split_strided` /
`..._split_2d`, `nvshmem_team_translate_pe`, `nvshmem_team_destroy`,
`nvshmemx_team_init`. See [`references/api/teams.md`](references/api/teams.md).

### Queue-Pair (QP) device APIs

Device-side QP control for fine-grained networking: `nvshmemx_qp_create`,
`nvshmemx_qp_put/get/p/g` (+ `_nbi`), `nvshmemx_qp_put_signal`,
`nvshmemx_qp_fence` / `..._quiet`. See [`references/api/qp.md`](references/api/qp.md).

### Environment variables (by category)

Full list with type / default / description: [`references/env.md`](references/env.md).

| Category | Common variables |
|----------|------------------|
| Standard | `NVSHMEM_SYMMETRIC_SIZE`, `NVSHMEM_DEBUG`, `NVSHMEM_INFO`, `NVSHMEM_VERSION` |
| Bootstrap | `NVSHMEM_BOOTSTRAP`, `NVSHMEM_BOOTSTRAP_PMI`, `NVSHMEM_BOOTSTRAP_PLUGIN` |
| Transport | `NVSHMEM_REMOTE_TRANSPORT`, `NVSHMEM_IB_ENABLE_IBGDA`, `NVSHMEM_DISABLE_P2P`, `NVSHMEM_HCA_LIST` |
| Collectives | `NVSHMEM_BCAST_ALGO`, `NVSHMEM_BARRIER_DISSEM_KVAL`, `NVSHMEM_REDMAXLOC_ALGO`, `NVSHMEM_FCOLLECT_LL_THRESHOLD` |
| NVTX / debug | `NVSHMEM_NVTX`, `NVSHMEM_DEBUG`, `NVSHMEM_DEBUG_FILE` |

### Python bindings (NVSHMEM4Py)

`nvshmem.core` host API (init/finalize, `nvshmem.array` symmetric allocation,
collectives, RMA, PyTorch & CuPy interop) plus device-side APIs for the
**Numba-CUDA** and **CuTe** DSLs (`put`/`get`, `barrier`/`broadcast`,
`atomic_add`, ...). Entry point:
[`references/nvshmem4py/index.md`](references/nvshmem4py/index.md);
overview & compatibility: [`references/nvshmem4py/overview.md`](references/nvshmem4py/overview.md);
device DSLs: [`references/nvshmem4py/device/`](references/nvshmem4py/device/).

## Troubleshooting quick links

| Symptom | Start here |
|---------|------------|
| Build / prerequisites / running fails | [`references/faq.md`](references/faq.md), [`references/using.md`](references/using.md) |
| Hang in put/get/collective, wrong results | [`references/memory-model.md`](references/memory-model.md) (ordering/atomicity), [`references/api/ordering.md`](references/api/ordering.md) |
| Stream/kernel deadlocks (circular deps) | [`references/cuda-model.md`](references/cuda-model.md) |
| MPI / OpenSHMEM interop | [`references/faq.md`](references/faq.md), [`references/using.md`](references/using.md) |
| GPU-GPU interconnect / IBGDA | [`references/faq.md`](references/faq.md), [`references/env.md`](references/env.md) |

## Search recipes

```bash
# A routine's full set of signatures (blocking/on_stream/block/warp variants)
rg -n -A6 "### \*\*NVSHMEM_PUT\*\*" references/api/rma.md

# Every page that mentions a concept
rg -rl "symmetric heap|active set|PGAS" references/

# An environment variable with its type and default
rg -n -A4 "^`NVSHMEM_IB_ENABLE_IBGDA`" references/env.md

# A reduction or collective across the API and examples
rg -n "fcollect|alltoall|broadcast" references/api/collectives.md references/examples.md

# NVSHMEM4Py method / device DSL usage
rg -n "nvshmem.core|barrier_all|atomic_add" references/nvshmem4py/

# Which page defines a routine
rg -n "NVSHMEMX_QP_PUT_SIGNAL" references/api/index.md
```

## Reading the NVSHMEM source

The docs describe behavior; the source explains it. Run
[`update-nvshmem.sh`](update-nvshmem.sh) to fetch the matching NVSHMEM source
(tag `v3.7.0-0`) into `repos/nvshmem/` for side-by-side reading:

```bash
nvshmem_skill/update-nvshmem.sh            # sparse: key dirs only (small)
nvshmem_skill/update-nvshmem.sh --full     # whole repo
```

Source map (`repos/nvshmem/`):

| Looking for | Source |
|-------------|--------|
| Public host / extension headers | `src/include/nvshmem.h`, `src/include/nvshmemx.h`, `src/include/nvshmem_host.h` |
| Device-side API (defines, coll, tile) | `src/include/device/` |
| Host RMA / AMO / signal / fence / quiet | `src/host/comm/` (`putget.cpp`, `amo.cpp`, `rma.cu`, `fence.cpp`, `quiet.cpp`, `sync.cpp`) |
| Host collectives | `src/host/coll/` (`barrier/`, `broadcast/`, `fcollect/`, `alltoall/`, `rdxn/`, `reducescatter/`) |
| Teams | `src/host/team/` |
| Init / bootstrap / topo | `src/host/init/`, `src/host/bootstrap/`, `src/host/topo/` |
| Transports (IB / IBGDA / proxy) | `src/host/transport/`, `src/host/proxy/`, `src/modules/transport/` |
| Device init / collective launch | `src/device/init/`, `src/device/launch/` |
| Python bindings (NVSHMEM4Py) | `nvshmem4py/nvshmem/core/`, `nvshmem4py/nvshmem/bindings/` |
| Runnable examples | `examples/` (`*.cu`, `hello.cpp`, `gemm_allreduce/`) |
| Micro-benchmarks | `perftest/` |

Cross-reference docs and code in one search, e.g.:

```bash
rg -n "nvshmem_put_signal" references/api/signal.md repos/nvshmem/src
rg -n "NVSHMEM_SYMMETRIC_SIZE" references/env.md repos/nvshmem/src
```

## Regenerating

These pages are scraped from the rendered docs site by
[`build_nvshmem_skill.py`](build_nvshmem_skill.py) (a `uv` script). Re-run it
after the docs change. See [`README.md`](README.md) for details.

```bash
nvshmem_skill/build_nvshmem_skill.py            # fetch + convert all pages
nvshmem_skill/build_nvshmem_skill.py --offline  # reuse cached HTML only
```

`SKILL.md`, `README.md`, and `references/INDEX.md` are hand-written and are not
overwritten by the build.
