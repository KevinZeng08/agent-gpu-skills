# NVSHMEM Device Collectives with CuTe DSL

This section documents the NVSHMEM Device Collective operations with CuTe DSL.

## Example: Using `barrier_all` and `broadcast` in a CuTe kernel

The following example demonstrates how to use the NVSHMEM `barrier_all` and `broadcast` collective operations in a CuTe kernel. Collectives enable synchronization and data movement among multiple PEs (processing elements) directly from device code.

    import cutlass
    from cutlass import cute
    from cuda.core import Device, Stream
    import nvshmem
    import nvshmem.core.device.cute as nvshmem_cute
    import nvshmem.core.interop.cute as nvshmem_cute_interop
    from mpi4py import MPI

    @cute.kernel
    def collective_kernel(buf: cute.Tensor, root: cutlass.Int32):
        # Synchronize all PEs at this point
        nvshmem_cute.barrier_all()
        # Broadcast buf from root PE to all PEs
        nvshmem_cute.broadcast_block(buf, buf, root)

    @cute.jit
    def collective_launcher(buf, root):
        collective_kernel[1, 1](buf, root)

    # Initialize NVSHMEM
    dev = Device()
    dev.set_current()
    stream = dev.create_stream()
    nvshmem.init(dev=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi", stream=stream)

    # Get information about the current PE
    me = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    # Set up buffer and root
    buf = nvshmem_cute_interop.tensor((1,), dtype=cute.Int32)
    root = 0

    # Compile and launch the kernel
    compiled_fn, nvshmem_kernel = nvshmem_cute_interop.cute_compile_helper(
        collective_launcher, buf, root
    )
    compiled_fn(buf, root, stream=stream)

    # Finalize NVSHMEM
    nvshmem.core.library_finalize(nvshmem_kernel)
    nvshmem_cute_interop.cleanup_cute()
    nvshmem.finalize(dev=dev, stream=stream)

This example synchronizes all PEs with `barrier_all` and then broadcasts the value in `buf` from the root PE (PE 0) to all PEs using the CTA-level `broadcast_block` operation.

`nvshmem.core.device.cute.collective. sync_block ( team )`

Executes a CTA-level synchronization across all PEs in `team`. All threads in the CTA must call this function.

This is a lightweight synchronization point that guarantees all PEs in the team have reached it before any PE proceeds. It does not provide memory-ordering or memory-visibility guarantees; use `barrier` when a memory fence is also required. All PEs in the team must call this function before any PE can proceed past it.

Args:

  * `team` (`int`): NVSHMEM team handle identifying the set of PEs to synchronize. Use `nvshmem.core.Teams.TEAM_WORLD` to synchronize all PEs.

Note:
    All PEs in `team` must call `sync_block` with the same `team` argument. Use `sync_all_block` to synchronize across all PEs without specifying a team.

`nvshmem.core.device.cute.collective. sync_warp ( team )`

Executes a warp-level synchronization across all PEs in `team`. All threads in the warp must call this function.

This is a lightweight synchronization point that guarantees all PEs in the team have reached it before any PE proceeds. It does not provide memory-ordering or memory-visibility guarantees; use `barrier` when a memory fence is also required. All PEs in the team must call this function before any PE can proceed past it.

Args:

  * `team` (`int`): NVSHMEM team handle identifying the set of PEs to synchronize. Use `nvshmem.core.Teams.TEAM_WORLD` to synchronize all PEs.

Note:
    All PEs in `team` must call `sync_warp` with the same `team` argument. Use `sync_all_warp` to synchronize across all PEs without specifying a team.

`nvshmem.core.device.cute.collective. sync ( team )`

Executes a thread-level synchronization across all PEs in `team`.

This is a lightweight synchronization point that guarantees all PEs in the team have reached it before any PE proceeds. It does not provide memory-ordering or memory-visibility guarantees; use `barrier` when a memory fence is also required. All PEs in the team must call this function before any PE can proceed past it.

Args:

  * `team` (`int`): NVSHMEM team handle identifying the set of PEs to synchronize. Use `nvshmem.core.Teams.TEAM_WORLD` to synchronize all PEs.

Note:
    All PEs in `team` must call `sync` with the same `team` argument. Use `sync_all` to synchronize across all PEs without specifying a team.

`nvshmem.core.device.cute.collective. sync_all ( )`

Executes a thread-level synchronization across all PEs in the NVSHMEM runtime (equivalent to `sync(TEAM_WORLD)`).

This is a convenience wrapper around `sync` that automatically uses `TEAM_WORLD` as the team, covering all PEs participating in the NVSHMEM job.

Note:
    All PEs must call `sync_all` before any PE can proceed past it.

`nvshmem.core.device.cute.collective. sync_all_block ( )`

Executes a CTA-level synchronization across all PEs in the NVSHMEM runtime (equivalent to `sync_block(TEAM_WORLD)`). All threads in the CTA must call this function.

This is a convenience wrapper around `sync_block` that automatically uses `TEAM_WORLD` as the team, covering all PEs participating in the NVSHMEM job.

Note:
    All PEs must call `sync_all_block` before any PE can proceed past it.

`nvshmem.core.device.cute.collective. sync_all_warp ( )`

Executes a warp-level synchronization across all PEs in the NVSHMEM runtime (equivalent to `sync_warp(TEAM_WORLD)`). All threads in the warp must call this function.

This is a convenience wrapper around `sync_warp` that automatically uses `TEAM_WORLD` as the team, covering all PEs participating in the NVSHMEM job.

Note:
    All PEs must call `sync_all_warp` before any PE can proceed past it.

`nvshmem.core.device.cute.collective. barrier ( team )`

Executes a thread-level barrier across all PEs in `team`.

A barrier combines synchronization with a full memory fence, ensuring that all outstanding NVSHMEM memory operations (puts, gets, atomics) issued before the barrier are complete and visible before any PE in the team proceeds past the barrier.

Args:

  * `team` (`int`): NVSHMEM team handle identifying the set of PEs to barrier on. Use `nvshmem.core.Teams.TEAM_WORLD` to barrier across all PEs.

Note:
    All PEs in `team` must call `barrier` with the same `team` argument. Use `barrier_all` to barrier across all PEs without specifying a team. `barrier` provides stronger ordering guarantees than `sync`.

`nvshmem.core.device.cute.collective. barrier_block ( team )`

Executes a CTA-level barrier across all PEs in `team`. All threads in the CTA must call this function.

A barrier combines synchronization with a full memory fence, ensuring that all outstanding NVSHMEM memory operations (puts, gets, atomics) issued before the barrier are complete and visible before any PE in the team proceeds past the barrier.

Args:

  * `team` (`int`): NVSHMEM team handle identifying the set of PEs to barrier on. Use `nvshmem.core.Teams.TEAM_WORLD` to barrier across all PEs.

Note:
    All PEs in `team` must call `barrier_block` with the same `team` argument. Use `barrier_all_block` to barrier across all PEs without specifying a team. `barrier` provides stronger ordering guarantees than `sync`.

`nvshmem.core.device.cute.collective. barrier_warp ( team )`

Executes a warp-level barrier across all PEs in `team`. All threads in the warp must call this function.

A barrier combines synchronization with a full memory fence, ensuring that all outstanding NVSHMEM memory operations (puts, gets, atomics) issued before the barrier are complete and visible before any PE in the team proceeds past the barrier.

Args:

  * `team` (`int`): NVSHMEM team handle identifying the set of PEs to barrier on. Use `nvshmem.core.Teams.TEAM_WORLD` to barrier across all PEs.

Note:
    All PEs in `team` must call `barrier_warp` with the same `team` argument. Use `barrier_all_warp` to barrier across all PEs without specifying a team. `barrier` provides stronger ordering guarantees than `sync`.

`nvshmem.core.device.cute.collective. barrier_all ( )`

Executes a thread-level barrier across all PEs in the NVSHMEM runtime (equivalent to `barrier(TEAM_WORLD)`).

This is a convenience wrapper around `barrier` that automatically uses `TEAM_WORLD` as the team. It combines synchronization with a full memory fence, ensuring all outstanding NVSHMEM memory operations are visible before proceeding.

Note:
    All PEs must call `barrier_all` before any PE can proceed past it. `barrier_all` provides stronger ordering guarantees than `sync_all`.

`nvshmem.core.device.cute.collective. barrier_all_block ( )`

Executes a CTA-level barrier across all PEs in the NVSHMEM runtime (equivalent to `barrier_block(TEAM_WORLD)`). All threads in the CTA must call this function.

This is a convenience wrapper around `barrier_block` that automatically uses `TEAM_WORLD` as the team. It combines synchronization with a full memory fence, ensuring all outstanding NVSHMEM memory operations are visible before proceeding.

Note:
    All PEs must call `barrier_all_block` before any PE can proceed past it. `barrier_all` provides stronger ordering guarantees than `sync_all`.

`nvshmem.core.device.cute.collective. barrier_all_warp ( )`

Executes a warp-level barrier across all PEs in the NVSHMEM runtime (equivalent to `barrier_warp(TEAM_WORLD)`). All threads in the warp must call this function.

This is a convenience wrapper around `barrier_warp` that automatically uses `TEAM_WORLD` as the team. It combines synchronization with a full memory fence, ensuring all outstanding NVSHMEM memory operations are visible before proceeding.

Note:
    All PEs must call `barrier_all_warp` before any PE can proceed past it. `barrier_all` provides stronger ordering guarantees than `sync_all`.

`nvshmem.core.device.cute.collective. reduce ( team , dst , src , op )`

Performs a thread-scoped all-reduce from `src` to `dst` across all PEs in `team`.

Each PE contributes `size(dst)` elements from `src`, and the result of applying the reduction operator `op` element-wise across all PEs is written to `dst` on every PE in the team (all-reduce semantics).

Args:

  * `team` (`int`): NVSHMEM team handle identifying the participating PEs.
  * `dst`: CuTe tensor view pointing to the symmetric destination array. Must be a symmetric (NVSHMEM-allocated) tensor. The number of elements reduced is `size(dst)`.
  * `src`: CuTe tensor view pointing to the symmetric source array. Must be a symmetric (NVSHMEM-allocated) tensor with at least `size(dst)` elements.
  * `op` (`str`): Reduction operator string. Supported operators for numeric types: `"sum"`, `"prod"`, `"min"`, `"max"`. Additional bitwise operators for integral types: `"and"`, `"or"`, `"xor"`.

Note:
    All PEs in `team` must call `reduce` before any PE can proceed past it. The element count is taken from `dst`. Passing an unsupported op/dtype combination raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.collective. reduce_block ( team , dst , src , op )`

Performs a CTA-scoped all-reduce from `src` to `dst` across all PEs in `team`. All threads in the CTA must call this function with the same arguments.

Each PE contributes `size(dst)` elements from `src`, and the result of applying the reduction operator `op` element-wise across all PEs is written to `dst` on every PE in the team (all-reduce semantics).

Args:

  * `team` (`int`): NVSHMEM team handle identifying the participating PEs.
  * `dst`: CuTe tensor view pointing to the symmetric destination array. Must be a symmetric (NVSHMEM-allocated) tensor. The number of elements reduced is `size(dst)`.
  * `src`: CuTe tensor view pointing to the symmetric source array. Must be a symmetric (NVSHMEM-allocated) tensor with at least `size(dst)` elements.
  * `op` (`str`): Reduction operator string. Supported operators for numeric types: `"sum"`, `"prod"`, `"min"`, `"max"`. Additional bitwise operators for integral types: `"and"`, `"or"`, `"xor"`.

Note:
    All PEs in `team` must call `reduce_block` before any PE can proceed past it. The element count is taken from `dst`. Passing an unsupported op/dtype combination raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.collective. reduce_warp ( team , dst , src , op )`

Performs a warp-scoped all-reduce from `src` to `dst` across all PEs in `team`. All threads in the warp must call this function with the same arguments.

Each PE contributes `size(dst)` elements from `src`, and the result of applying the reduction operator `op` element-wise across all PEs is written to `dst` on every PE in the team (all-reduce semantics).

Args:

  * `team` (`int`): NVSHMEM team handle identifying the participating PEs.
  * `dst`: CuTe tensor view pointing to the symmetric destination array. Must be a symmetric (NVSHMEM-allocated) tensor. The number of elements reduced is `size(dst)`.
  * `src`: CuTe tensor view pointing to the symmetric source array. Must be a symmetric (NVSHMEM-allocated) tensor with at least `size(dst)` elements.
  * `op` (`str`): Reduction operator string. Supported operators for numeric types: `"sum"`, `"prod"`, `"min"`, `"max"`. Additional bitwise operators for integral types: `"and"`, `"or"`, `"xor"`.

Note:
    All PEs in `team` must call `reduce_warp` before any PE can proceed past it. The element count is taken from `dst`. Passing an unsupported op/dtype combination raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.collective. reducescatter ( team , dst , src , op )`

Performs a thread-scoped reduce-scatter from `src` to `dst` across all PEs in `team`.

In a reduce-scatter, each PE contributes elements from `src`, and the result of applying the reduction operator element-wise across all PEs is divided into equal portions, each portion written to `dst` on a different PE (scatter semantics).

Args:

  * `team` (`int`): NVSHMEM team handle identifying the participating PEs.
  * `dst`: CuTe tensor view pointing to the symmetric destination array. Must be a symmetric (NVSHMEM-allocated) tensor. The number of output elements per PE is `size(dst)`. The `src` array should have at least `size(dst) * team_n_pes(team)` elements.
  * `src`: CuTe tensor view pointing to the symmetric source array. Must be a symmetric (NVSHMEM-allocated) tensor.
  * `op` (`str`): Reduction operator string. Supported operators for numeric types: `"sum"`, `"prod"`, `"min"`, `"max"`. Additional bitwise operators for integral types: `"and"`, `"or"`, `"xor"`.

Note:
    All PEs in `team` must call `reducescatter` before any PE can proceed past it. The element count is taken from `dst`. Passing an unsupported op/dtype combination raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.collective. reducescatter_block ( team , dst , src , op )`

Performs a CTA-scoped reduce-scatter from `src` to `dst` across all PEs in `team`. All threads in the CTA must call this function with the same arguments.

In a reduce-scatter, each PE contributes elements from `src`, and the result of applying the reduction operator element-wise across all PEs is divided into equal portions, each portion written to `dst` on a different PE (scatter semantics).

Args:

  * `team` (`int`): NVSHMEM team handle identifying the participating PEs.
  * `dst`: CuTe tensor view pointing to the symmetric destination array. Must be a symmetric (NVSHMEM-allocated) tensor. The number of output elements per PE is `size(dst)`. The `src` array should have at least `size(dst) * team_n_pes(team)` elements.
  * `src`: CuTe tensor view pointing to the symmetric source array. Must be a symmetric (NVSHMEM-allocated) tensor.
  * `op` (`str`): Reduction operator string. Supported operators for numeric types: `"sum"`, `"prod"`, `"min"`, `"max"`. Additional bitwise operators for integral types: `"and"`, `"or"`, `"xor"`.

Note:
    All PEs in `team` must call `reducescatter_block` before any PE can proceed past it. The element count is taken from `dst`. Passing an unsupported op/dtype combination raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.collective. reducescatter_warp ( team , dst , src , op )`

Performs a warp-scoped reduce-scatter from `src` to `dst` across all PEs in `team`. All threads in the warp must call this function with the same arguments.

In a reduce-scatter, each PE contributes elements from `src`, and the result of applying the reduction operator element-wise across all PEs is divided into equal portions, each portion written to `dst` on a different PE (scatter semantics).

Args:

  * `team` (`int`): NVSHMEM team handle identifying the participating PEs.
  * `dst`: CuTe tensor view pointing to the symmetric destination array. Must be a symmetric (NVSHMEM-allocated) tensor. The number of output elements per PE is `size(dst)`. The `src` array should have at least `size(dst) * team_n_pes(team)` elements.
  * `src`: CuTe tensor view pointing to the symmetric source array. Must be a symmetric (NVSHMEM-allocated) tensor.
  * `op` (`str`): Reduction operator string. Supported operators for numeric types: `"sum"`, `"prod"`, `"min"`, `"max"`. Additional bitwise operators for integral types: `"and"`, `"or"`, `"xor"`.

Note:
    All PEs in `team` must call `reducescatter_warp` before any PE can proceed past it. The element count is taken from `dst`. Passing an unsupported op/dtype combination raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.collective. fcollect ( team , dst , src )`

Performs a thread-scoped fcollect (all-gather) from `src` to `dst` across all PEs in `team`.

Each PE contributes `size(src)` elements, and the concatenated result from all PEs is written to `dst` on every PE. The `dst` array must be large enough to hold contributions from all PEs: `size(dst) >= size(src) * team_n_pes(team)`.

Args:

  * `team` (`int`): NVSHMEM team handle identifying the participating PEs.
  * `dst`: CuTe tensor view pointing to the symmetric destination array on this PE. Must be a symmetric (NVSHMEM-allocated) tensor with at least `size(src) * team_n_pes(team)` elements.
  * `src`: CuTe tensor view pointing to the symmetric source array on this PE. Must be a symmetric (NVSHMEM-allocated) tensor. The number of elements contributed per PE is `size(src)`.

Note:
    All PEs in `team` must call `fcollect` before any PE can proceed past it. The element count per PE is taken from `src`. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.collective. fcollect_block ( team , dst , src )`

Performs a CTA-scoped fcollect (all-gather) from `src` to `dst` across all PEs in `team`. All threads in the CTA must call this function with the same arguments.

Each PE contributes `size(src)` elements, and the concatenated result from all PEs is written to `dst` on every PE. The `dst` array must be large enough to hold contributions from all PEs: `size(dst) >= size(src) * team_n_pes(team)`.

Args:

  * `team` (`int`): NVSHMEM team handle identifying the participating PEs.
  * `dst`: CuTe tensor view pointing to the symmetric destination array on this PE. Must be a symmetric (NVSHMEM-allocated) tensor with at least `size(src) * team_n_pes(team)` elements.
  * `src`: CuTe tensor view pointing to the symmetric source array on this PE. Must be a symmetric (NVSHMEM-allocated) tensor. The number of elements contributed per PE is `size(src)`.

Note:
    All PEs in `team` must call `fcollect_block` before any PE can proceed past it. The element count per PE is taken from `src`. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.collective. fcollect_warp ( team , dst , src )`

Performs a warp-scoped fcollect (all-gather) from `src` to `dst` across all PEs in `team`. All threads in the warp must call this function with the same arguments.

Each PE contributes `size(src)` elements, and the concatenated result from all PEs is written to `dst` on every PE. The `dst` array must be large enough to hold contributions from all PEs: `size(dst) >= size(src) * team_n_pes(team)`.

Args:

  * `team` (`int`): NVSHMEM team handle identifying the participating PEs.
  * `dst`: CuTe tensor view pointing to the symmetric destination array on this PE. Must be a symmetric (NVSHMEM-allocated) tensor with at least `size(src) * team_n_pes(team)` elements.
  * `src`: CuTe tensor view pointing to the symmetric source array on this PE. Must be a symmetric (NVSHMEM-allocated) tensor. The number of elements contributed per PE is `size(src)`.

Note:
    All PEs in `team` must call `fcollect_warp` before any PE can proceed past it. The element count per PE is taken from `src`. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.collective. broadcast ( team , dst , src , root=0 )`

Performs a thread-scoped broadcast from `src` on `root` PE to `dst` on all PEs in `team`.

The root PE broadcasts the contents of its `src` array to the `dst` array on every PE in the team (including the root itself). Non-root PEs ignore their own `src`.

Args:

  * `team` (`int`): NVSHMEM team handle identifying the participating PEs.
  * `dst`: CuTe tensor view pointing to the symmetric destination array. Must be a symmetric (NVSHMEM-allocated) tensor on all PEs.
  * `src`: CuTe tensor view pointing to the symmetric source array on the root PE. Must be a symmetric (NVSHMEM-allocated) tensor. Only the root PE’s `src` is used.
  * `root` (`int`, optional): PE rank within `team` that serves as the broadcast source. Defaults to `0` (the first PE in the team).

Note:
    All PEs in `team` must call `broadcast` before any PE can proceed past it. The number of elements broadcast is `min(size(dst), size(src))`. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.collective. broadcast_block ( team , dst , src , root=0 )`

Performs a CTA-scoped broadcast from `src` on `root` PE to `dst` on all PEs in `team`. All threads in the CTA must call this function with the same arguments.

The root PE broadcasts the contents of its `src` array to the `dst` array on every PE in the team (including the root itself). Non-root PEs ignore their own `src`.

Args:

  * `team` (`int`): NVSHMEM team handle identifying the participating PEs.
  * `dst`: CuTe tensor view pointing to the symmetric destination array. Must be a symmetric (NVSHMEM-allocated) tensor on all PEs.
  * `src`: CuTe tensor view pointing to the symmetric source array on the root PE. Must be a symmetric (NVSHMEM-allocated) tensor. Only the root PE’s `src` is used.
  * `root` (`int`, optional): PE rank within `team` that serves as the broadcast source. Defaults to `0` (the first PE in the team).

Note:
    All PEs in `team` must call `broadcast_block` before any PE can proceed past it. The number of elements broadcast is `min(size(dst), size(src))`. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.collective. broadcast_warp ( team , dst , src , root=0 )`

Performs a warp-scoped broadcast from `src` on `root` PE to `dst` on all PEs in `team`. All threads in the warp must call this function with the same arguments.

The root PE broadcasts the contents of its `src` array to the `dst` array on every PE in the team (including the root itself). Non-root PEs ignore their own `src`.

Args:

  * `team` (`int`): NVSHMEM team handle identifying the participating PEs.
  * `dst`: CuTe tensor view pointing to the symmetric destination array. Must be a symmetric (NVSHMEM-allocated) tensor on all PEs.
  * `src`: CuTe tensor view pointing to the symmetric source array on the root PE. Must be a symmetric (NVSHMEM-allocated) tensor. Only the root PE’s `src` is used.
  * `root` (`int`, optional): PE rank within `team` that serves as the broadcast source. Defaults to `0` (the first PE in the team).

Note:
    All PEs in `team` must call `broadcast_warp` before any PE can proceed past it. The number of elements broadcast is `min(size(dst), size(src))`. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.collective. alltoall ( team , dst , src )`

Performs a thread-scoped all-to-all exchange from `src` to `dst` across all PEs in `team`.

In an all-to-all operation, each PE sends a distinct portion of its `src` array to every other PE, and receives a portion from every PE into its `dst` array. The `src` array is logically divided into `team_n_pes(team)` equal segments; segment `i` is sent to PE `i`. Each PE receives one segment from every PE into the corresponding portion of `dst`.

Args:

  * `team` (`int`): NVSHMEM team handle identifying the participating PEs.
  * `dst`: CuTe tensor view pointing to the symmetric destination array on this PE. Must be a symmetric (NVSHMEM-allocated) tensor. Must have at least `size(src)` elements (same total size as `src`).
  * `src`: CuTe tensor view pointing to the symmetric source array on this PE. Must be a symmetric (NVSHMEM-allocated) tensor. The per-PE send count is `size(src) // team_n_pes(team)` elements.

Note:
    All PEs in `team` must call `alltoall` before any PE can proceed past it. `size(src)` must be evenly divisible by `team_n_pes(team)`. The per-PE element count passed to NVSHMEM is `size(src) // team_n_pes(team)`. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.collective. alltoall_block ( team , dst , src )`

Performs a CTA-scoped all-to-all exchange from `src` to `dst` across all PEs in `team`. All threads in the CTA must call this function with the same arguments.

In an all-to-all operation, each PE sends a distinct portion of its `src` array to every other PE, and receives a portion from every PE into its `dst` array. The `src` array is logically divided into `team_n_pes(team)` equal segments; segment `i` is sent to PE `i`. Each PE receives one segment from every PE into the corresponding portion of `dst`.

Args:

  * `team` (`int`): NVSHMEM team handle identifying the participating PEs.
  * `dst`: CuTe tensor view pointing to the symmetric destination array on this PE. Must be a symmetric (NVSHMEM-allocated) tensor. Must have at least `size(src)` elements (same total size as `src`).
  * `src`: CuTe tensor view pointing to the symmetric source array on this PE. Must be a symmetric (NVSHMEM-allocated) tensor. The per-PE send count is `size(src) // team_n_pes(team)` elements.

Note:
    All PEs in `team` must call `alltoall_block` before any PE can proceed past it. `size(src)` must be evenly divisible by `team_n_pes(team)`. The per-PE element count passed to NVSHMEM is `size(src) // team_n_pes(team)`. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.

`nvshmem.core.device.cute.collective. alltoall_warp ( team , dst , src )`

Performs a warp-scoped all-to-all exchange from `src` to `dst` across all PEs in `team`. All threads in the warp must call this function with the same arguments.

In an all-to-all operation, each PE sends a distinct portion of its `src` array to every other PE, and receives a portion from every PE into its `dst` array. The `src` array is logically divided into `team_n_pes(team)` equal segments; segment `i` is sent to PE `i`. Each PE receives one segment from every PE into the corresponding portion of `dst`.

Args:

  * `team` (`int`): NVSHMEM team handle identifying the participating PEs.
  * `dst`: CuTe tensor view pointing to the symmetric destination array on this PE. Must be a symmetric (NVSHMEM-allocated) tensor. Must have at least `size(src)` elements (same total size as `src`).
  * `src`: CuTe tensor view pointing to the symmetric source array on this PE. Must be a symmetric (NVSHMEM-allocated) tensor. The per-PE send count is `size(src) // team_n_pes(team)` elements.

Note:
    All PEs in `team` must call `alltoall_warp` before any PE can proceed past it. `size(src)` must be evenly divisible by `team_n_pes(team)`. The per-PE element count passed to NVSHMEM is `size(src) // team_n_pes(team)`. Passing an unsupported dtype raises `RuntimeError` at JIT compile time.
