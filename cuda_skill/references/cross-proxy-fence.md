# Cross-Proxy Fence: GPU Memory Ordering Between CUDA Cores and Async Co-processors

## Quick Reference

When CUDA Core (generic proxy) and async hardware (TMA/Tensor Core) access the same SMEM/TMEM,
you need **cross-proxy fences** in addition to normal release-acquire synchronization.

| Direction | State Space | Fence Needed | PTX | SASS | CUTLASS Helper |
|-----------|------------|--------------|-----|------|----------------|
| CUDA Core → TMA | SMEM | `fence.proxy.async.shared::cta` | Explicit | `FENCE.VIEW.ASYNC.S` | `fence_view_async_shared()` |
| CUDA Core → tcgen05 | SMEM | `fence.proxy.async.shared::cta` | Explicit | `FENCE.VIEW.ASYNC.S` | `fence_view_async_shared()` |
| CUDA Core → tcgen05 | TMEM (write) | `tcgen05.wait::st` | Explicit | `FENCE.VIEW.ASYNC.T` | `fence_view_async_tmem_store()` |
| CUDA Core → tcgen05 | TMEM (WAR) | `tcgen05.wait::ld` | Explicit | `FENCE.VIEW.ASYNC.T` | `fence_view_async_tmem_load()` |
| TMA → CUDA Core | SMEM | **Implicit** in TMA completion | Not needed | N/A | N/A |
| tcgen05 → CUDA Core | TMEM | **Implicit** in `tcgen05.commit` | Not needed | `UTCBAR` | N/A |
| TMA → tcgen05 | SMEM | **Implicit** in TMA completion | Not needed | N/A | N/A |

**Key asymmetry: Async→Generic is implicit. Generic→Async is explicit and easy to forget.**

---

## 1. Core Concepts

### 1.1. What is a Proxy?

A **memory proxy** is an abstract label for a method of memory access (PTX ISA 8.6).

- **Generic proxy**: Memory operations issued by CUDA cores (`ld`, `st`, `atom`, `red`).
- **Async proxy**: Memory operations issued by asynchronous co-processors (TMA, wgmma, tcgen05, mbarrier).

Memory ordering operations (fences, release-acquire) that work within the generic proxy
**do NOT automatically extend to the async proxy**. A cross-proxy fence bridges the two.

### 1.2. Execution Order vs Memory Order

Execution order (e.g., via mbarrier arrive/wait) guarantees instructions execute in sequence.
Memory order guarantees that memory side-effects are **visible** across threads/proxies.

```
// WRONG: execution order alone is NOT sufficient
thread A:
    st.shared addr, val
    mbarrier.arrive.relaxed        // only execution order, no memory order
thread B:
    mbarrier.try_wait.relaxed
    ld.shared reg, addr            // may read stale value!
```

You always need **both** execution order AND memory order (release-acquire) for correct
producer-consumer relationships.

### 1.3. Three Dimensions of Synchronization

Every synchronization decision involves three orthogonal dimensions:

1. **Scope**: `.cta` / `.cluster` / `.gpu` / `.sys` — which threads participate
2. **State space**: `.shared::cta` (SMEM) / `.shared::cluster` (DSMEM) / `.global` (GMEM) — which memory
3. **Proxy**: generic / async — which hardware unit

Use the weakest (cheapest) option sufficient for correctness along each dimension.

---

## 2. The Four Cross-Proxy Patterns

### 2.1. Generic → Async (CUDA Core → TMA/Tensor Core)

**This is where bugs live.** You must insert an explicit proxy fence.

The proxy fence "ties" the async proxy's view of memory to the generic proxy's release-acquire
pattern. Without it, `mbarrier.arrive.release` only makes data visible in the generic proxy.

#### Pattern: CUDA Core writes SMEM, TMA reads SMEM

```
Producer threads:
    st.shared addr, reg                    // generic proxy write to SMEM
    fence.proxy.async.shared::cta          // <-- REQUIRED: make SMEM visible to async proxy
    mbarrier.arrive                        // release (memory order in generic proxy)

Consumer threads:
    mbarrier.try_wait                      // acquire
    # tma store
    if thread0:
        cp.async.bulk.tensor.3d.global.shared::cta  // TMA reads SMEM (async proxy)
```

#### Pattern: CUDA Core writes SMEM, tcgen05 reads SMEM

Same as above — `fence.proxy.async.shared::cta` before mbarrier.arrive.

#### Pattern: CUDA Core writes TMEM, tcgen05 reads TMEM

```
Producer threads:
    tcgen05.st addr, reg                   // generic proxy write to TMEM
    tcgen05.wait::st                       // <-- REQUIRED: make TMEM visible to async proxy
    mbarrier.arrive                        // release

Consumer threads:
    mbarrier.try_wait                      // acquire
    if thread0:
        tcgen05.mma addr, A, B, C          // async proxy reads TMEM
```

#### Pattern: CUDA Core reads TMEM then overwrites same address (WAR hazard)

TMEM has no hardware dependency tracking. If `tcgen05.ld` reads an address and a
subsequent `tcgen05.st` writes the same address, you must insert `tcgen05.wait::ld`
to ensure the load completes before the store begins:

```
tcgen05.ld reg, addr                       // async read from TMEM
tcgen05.wait::ld                           // <-- REQUIRED: WAR hazard protection
tcgen05.st addr, new_reg                   // async write to same TMEM address
```

**Note**: When `tcgen05.ld` feeds a register to a subsequent CUDA Core instruction
(e.g., `add reg, reg, 1`) in the **same thread**, `tcgen05.wait::ld` is NOT needed —
the register dependency already enforces the correct ordering. But this register
dependency does NOT extend to memory ordering, so `tcgen05.wait::ld` is still required
if the same TMEM address will be written afterwards.

### 2.2. Async → Generic (TMA/Tensor Core → CUDA Core)

**Implicit — no explicit fence needed** in most cases.

TMA completion and `tcgen05.commit` both implicitly:
1. Arrive on the mbarrier with **release** semantic
2. Issue an **implicit async proxy fence** tying the async write to the release

#### TMA → CUDA Core

```
Producer threads:
    if thread0:
        cp.async.bulk.tensor.3d.shared::cta.global addr    // TMA writes SMEM

Consumer threads:
    mbarrier.try_wait                      // acquire — SMEM is already visible
    ld.shared reg, addr                    // safe: implicit proxy fence in TMA completion
```

#### tcgen05 → CUDA Core

`tcgen05.commit` explicitly tracks `tcgen05.mma` completion and implicitly arrives on
the mbarrier with release semantic + async proxy fence. SASS: `UTCBAR`.

```
Producer threads:
    if thread0:
        tcgen05.mma addr, A, B, C          // async proxy writes TMEM
        tcgen05.commit                      // arrives on mbarrier + implicit proxy fence

Consumer threads:
    mbarrier.try_wait                      // acquire — TMEM is already visible
    tcgen05.ld reg, addr                   // safe
    add reg, reg, 1                        // CUDA Core consumes TMEM data
```

### 2.3. Async → Async (TMA → tcgen05)

**Also implicit.** TMA completion's implicit proxy fence makes SMEM visible to both
generic and async proxies when the mbarrier phase completes.

```
Producer threads:
    if thread0:
        cp.async.bulk.tensor.3d.shared::cta.global addr

Consumer threads:
    mbarrier.try_wait                      // acquire — SMEM visible to all proxies
    if thread0:
        tcgen05.mma addr, A, B, C          // safe
```

### 2.4. Generic → Generic (CUDA Core → CUDA Core)

No proxy fence needed — standard fences suffice:

| Scope | Fence | SASS |
|-------|-------|------|
| `.cta` | `fence.cta` / `__syncthreads()` / `__threadfence_block()` | `MEMBAR.CTA` |
| `.cluster` | `fence.cluster` / `barrier.cluster` | `MEMBAR.GPU` |
| `.gpu` | `fence.gpu` / `__threadfence()` | `MEMBAR.GPU` |

---

## 3. Real-World Bug: cuLA CUDA Core → TMA Race ([PR #77](https://github.com/inclusionAI/cuLA/pull/77))

### 3.1. The Bug

In cuLA's warp-specialized attention kernel, the **prologue warp group** (CUDA Cores) processes
data in SMEM pipeline buffers, then releases buffers back to the **TMA load warp** for reuse:

```
// Prologue warp (CUDA Core, generic proxy):
k_pipeline.consumer_wait(k_pipe_state_read);    // acquire: TMA data visible ✓
// ... read K from SMEM, compute K_proc, write results back to SMEM via st.shared ...
*reinterpret_cast<bf16x8*>(&sK_dst(y, t)) = out;  // generic proxy write
k_pipeline.consumer_release(k_pipe_state_read);    // signals TMA: buffer free

// TMA load warp (async proxy):
// ... sees consumer_release, starts cp.async.bulk to same SMEM buffer ...
```

**Problem**: `consumer_release` contains `mbarrier.arrive` with release semantic, but this only
makes the CUDA Core's SMEM writes visible in the **generic proxy**. The TMA unit (async proxy)
may not see the writes, causing:

- TMA could start overwriting the buffer while CUDA Core writes are still in the LSU pipeline
- **Non-deterministic output** — requires 100K+ iterations to reproduce
- WAW (write-after-write) hazard between generic proxy and async proxy on the same SMEM location

### 3.2. The Fix

Insert `fence_view_async_shared()` (which emits `fence.proxy.async.shared::cta`)
**before** `consumer_release`:

```cpp
// CORRECT: proxy fence before releasing buffer to TMA
fence_view_async_shared();                         // make SMEM visible to async proxy
k_pipeline.consumer_release(k_pipe_state_read);    // now TMA can safely reuse buffer
```

### 3.3. Why the Reverse Direction (TMA → CUDA Core) Needs No Fence

An earlier commit incorrectly added `fence_view_async_shared()` **after** `consumer_wait`:

```cpp
// WRONG: redundant fence — TMA completion already has implicit proxy fence
k_pipeline.consumer_wait(k_pipe_state_read);
fence_view_async_shared();  // unnecessary!
```

This was removed because TMA's `cp.async.bulk` completion implicitly includes an async-generic
proxy fence. When `consumer_wait` (mbarrier.try_wait) succeeds, SMEM is already visible to
the generic proxy.

### 3.4. Timeline of Commits

| Commit | Action | Correct? |
|--------|--------|----------|
| `4cf525e` | Added fence after `consumer_wait` (TMA→Core direction) | Redundant but harmless |
| `14bcc17` | Removed redundant TMA→Core fences | Correct |
| `0d876b2` | Added fence before `consumer_release` (Core→TMA direction) | **The actual fix** |

---

## 4. Pipeline Buffer Lifecycle and Fence Placement

In warp-specialized kernels with TMA pipelines, each SMEM buffer goes through this lifecycle:

```
┌─────────────────────────────────────────────────────────────────────┐
│ TMA Load Warp                    Compute Warp                      │
│                                                                     │
│ producer_acquire()               ← waits for free buffer            │
│ cp.async.bulk (TMA write)                                          │
│ producer_commit()                                                   │
│   └─ mbarrier.arrive(release)                                      │
│   └─ implicit proxy fence ──────→ consumer_wait()                  │
│                                    └─ mbarrier.try_wait(acquire)   │
│                                  ld.shared / st.shared (CUDA Core) │
│                                  fence.proxy.async.shared::cta ◀── REQUIRED HERE │
│                                  consumer_release()                │
│   ┌─ mbarrier arrives ──────────   └─ mbarrier.arrive(release)     │
│   ↓                                                                 │
│ producer_acquire() (next iter)                                     │
│ cp.async.bulk (TMA write)                                          │
│ ...                                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Rule**: Insert `fence.proxy.async.shared::cta` (or `fence_view_async_shared()` in CUTLASS)
at the transition point where SMEM ownership passes from generic proxy back to async proxy.

---

## 5. Checklist: When Do You Need a Cross-Proxy Fence?

Ask these questions for every SMEM/TMEM pipeline buffer transition:

### Must insert `fence.proxy.async.shared::cta`:

- [ ] CUDA Core wrote to SMEM, and TMA will next read or write the same SMEM
- [ ] CUDA Core wrote to SMEM, and tcgen05 will next read the same SMEM
- [ ] CUDA Core wrote to SMEM, and `consumer_release()` returns buffer to TMA pipeline
- [ ] CUDA Core wrote to SMEM, and `producer_commit()` signals data ready for tcgen05

### Must insert `tcgen05.wait::st` (`fence_view_async_tmem_store()`):

- [ ] CUDA Core wrote to TMEM via `tcgen05.st`, and tcgen05 will next read the same TMEM

### Must insert `tcgen05.wait::ld` (`fence_view_async_tmem_load()`):

- [ ] CUDA Core read from TMEM via `tcgen05.ld`, and will next write the same TMEM address
  (WAR hazard — no HW dependency tracking for TMEM)
- [ ] Note: register dependency (`tcgen05.ld` → same-thread ALU op) does NOT require this fence

### No fence needed:

- [ ] TMA wrote to SMEM, CUDA Core reads it after `mbarrier.try_wait` — implicit proxy fence
- [ ] TMA wrote to SMEM, tcgen05 reads it after `mbarrier.try_wait` — implicit proxy fence
- [ ] tcgen05 wrote to TMEM via `tcgen05.commit`, CUDA Core reads after `mbarrier.try_wait` — implicit
- [ ] CUDA Core → CUDA Core (same proxy) — normal fences suffice

---

## 6. CUTLASS / CuTe API Reference

### fence_view_async_shared()

```cpp
// Definition: cutlass/arch/barrier.h:728
CUTLASS_DEVICE void fence_view_async_shared() {
    asm volatile (
        "fence.proxy.async.shared::cta;"
        ::: "memory");
}
```

Emits `FENCE.VIEW.ASYNC.S` in SASS. Use before any pipeline `consumer_release()` or
`producer_commit()` when CUDA Cores have written to the SMEM buffer.

### fence_view_async_tmem_store()

```cpp
// Definition: cutlass/arch/barrier.h:936
CUTE_DEVICE static void fence_view_async_tmem_store() {
    asm volatile (
        "tcgen05.wait::st.sync.aligned;"
        ::: "memory");
}
```

Emits `FENCE.VIEW.ASYNC.T` in SASS. Use after `tcgen05.st` (CUDA Core writing to TMEM)
before signaling the tcgen05 async proxy to consume the TMEM data. Ensures TMEM writes
from the generic proxy are visible to the async proxy.

**Use case**: CUDA Core corrects the accumulator (e.g., `Acc = Acc * alpha`) in TMEM
via `tcgen05.st`, then tcgen05.mma needs to read the corrected accumulator.

### fence_view_async_tmem_load()

```cpp
// Definition: cutlass/arch/barrier.h:923
CUTE_DEVICE static void fence_view_async_tmem_load() {
    asm volatile (
        "tcgen05.wait::ld.sync.aligned;"
        ::: "memory");
}
```

Emits `FENCE.VIEW.ASYNC.T` in SASS. Use after `tcgen05.ld` when the same TMEM address
will subsequently be written (WAR hazard protection). Since TMEM has no hardware
dependency tracking, this fence ensures the asynchronous load completes before any
subsequent write to the same address.

**Use case**: CUDA Core reads accumulator from TMEM via `tcgen05.ld` for epilogue
computation, then later stores a new value back to the same TMEM address.

### Where CUTLASS uses it

- **Epilogue TMA store**: After CUDA Core writes results to SMEM, before TMA stores to GMEM
- **Pipeline consumer_release**: Before releasing SMEM buffer back to TMA producer
- **GEMM mainloop**: Before signaling MMA warp that SMEM data is ready (when CUDA Core produced it)
- **FMHA kernels**: Before tcgen05.mma consumes SMEM data written by CUDA Core (e.g., softmax P matrix)
- **TMEM accumulator correction**: `fence_view_async_tmem_store()` after CUDA Core writes corrected accumulator to TMEM
- **TMEM epilogue readback**: `fence_view_async_tmem_load()` after CUDA Core reads TMEM via `tcgen05.ld` before overwriting

---

## 7. Performance Notes

| Fence | SASS | Relative Cost |
|-------|------|---------------|
| `fence.proxy.async.shared::cta` | `FENCE.VIEW.ASYNC.S` | Very cheap — only orders SMEM proxy view |
| `tcgen05.wait::st` / `tcgen05.wait::ld` | `FENCE.VIEW.ASYNC.T` | Very cheap — only orders TMEM proxy view |
| `fence.proxy.async` (all state spaces) | `FENCE.VIEW.ASYNC` | Slightly more expensive |
| `fence.cta` / `__syncthreads()` | `MEMBAR.CTA` | Moderate — orders all generic proxy ops |
| `fence.cluster` | `MEMBAR.GPU` | Expensive — GPU-scope memory barrier |
| `fence.gpu` / `__threadfence()` | `MEMBAR.GPU` | Expensive — GPU-scope memory barrier |

Always use the **narrowest state space** qualifier possible:
- `fence.proxy.async.shared::cta` > `fence.proxy.async` (prefer the former for SMEM-only)
- `sync_restrict::shared::cluster` avoids `MEMBAR.GPU` for cluster-scope DSMEM operations

---

## 8. Debugging Tips

### Symptoms of missing cross-proxy fence:
- **Non-deterministic results** that require many iterations (10K+) to reproduce
- Results are correct with `__syncthreads()` but incorrect with mbarrier-based pipelines
- Bug disappears when adding `printf` or `__nanosleep` (timing-dependent)
- `compute-sanitizer --tool racecheck` may flag shared memory races

### Diagnostic approach:
1. Add `fence_view_async_shared()` before every `consumer_release()` / `producer_commit()`
2. Add `fence_view_async_tmem_store()` after every `tcgen05.st` sequence
3. Add `fence_view_async_tmem_load()` after every `tcgen05.ld` that precedes a write to same address
4. If bug disappears, remove fences one by one to find the missing one
5. For each SMEM/TMEM buffer, trace the full lifecycle: who writes (which proxy), who reads next (which proxy)

---

## References

- [PTX ISA 9.1 - Memory Consistency Model: Proxies](ptx-docs/8-memory-consistency-model/8.6-proxies.md)
- [PTX ISA 9.1 - fence/membar](ptx-docs/9-instruction-set/9.7.13.4-parallel-synchronization-and-communication-instructionsmembarfence.md)
- [PTX ISA 9.1 - Async Proxy (cp.async.bulk)](ptx-docs/9-instruction-set/9.7.9.25-data-movement-and-conversion-instructions-asynchronous-copy.md)
- [PTX ISA 9.1 - Async Proxy (wgmma)](ptx-docs/9-instruction-set/9.7.15.4-async-proxy.md)
- [PTX ISA 9.1 - tcgen05 Memory Consistency](ptx-docs/9-instruction-set/9.7.16.6-memory-consistency-model-for-5th-generation-of-tensorcore-operations.md)
- [Yang Yifan - GPU Memory Consistency Model (Blog)](https://yang-yifan.github.io/blogs/memory_model/memory_model.html)
- [CUTLASS barrier.h - fence_view_async_shared()](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/barrier.h)
