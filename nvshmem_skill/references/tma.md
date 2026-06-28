# Using TMA with NVSHMEM

Tensor Memory Accelerator (TMA) is a CUDA hardware mechanism for issuing bulk asynchronous copies with low thread overhead. NVSHMEM can use TMA for device-side point-to-point transfers when the transfer is between GPUs that are directly reachable through the GPU load/store path, such as over NVLink.

TMA support does not add a separate TMA-specific put or get API. Applications continue to call the existing NVSHMEM put, get, and put-with-signal routines. When the runtime policy, GPU architecture, shared-memory registration, topology, and transfer shape all allow TMA, NVSHMEM routes the operation through the TMA path. Otherwise, the operation preserves normal NVSHMEM semantics by using the regular implementation.

## Motivation

TMA is useful when an application needs high NVLink bandwidth but cannot afford to dedicate many CUDA threads to communication. For large transfers, TMA can drive NVLink efficiently with very few issuing threads, including one issuing thread per CTA in the TMA path. This is especially useful in fused kernels where most CTA resources are reserved for computation.

TMA also makes NVSHMEM’s nonblocking, unordered put path over NVLink more useful. A TMA-backed nonblocking put can enqueue the transfer and return before the remote write is visible, allowing the kernel to overlap communication with other work. Applications can use `nvshmemx_flush` to wait only until a local source buffer is safe to reuse, and `nvshmem_quiet` when the remote PE must observe the data or when normal NVSHMEM ordering and completion are required.

The shared-memory operand required by TMA is also a natural fit for fused GPU kernels that produce tiles in shared memory. In that case, NVSHMEM can transfer the shared-memory tile directly to peer global memory without first copying the tile through local global memory.

## Enabling TMA

TMA is disabled by default while the feature is experimental. Enable it by setting the `NVSHMEM_TMA_POLICY` environment variable before NVSHMEM initialization:

    export NVSHMEM_TMA_POLICY=ENABLE

The supported policy values are:

`DISABLE`
    Do not use TMA. This is the default.
`ENABLE`
    Allow NVSHMEM to use TMA opportunistically for transfers that satisfy the per-transfer TMA routing requirements. If a particular transfer is not TMA-eligible, NVSHMEM uses the regular path.
`FORCE`
    Require TMA support to be available during initialization. Initialization fails on devices that do not support TMA. `FORCE` does not force every put or get onto the TMA path; individual transfers still follow the same per-transfer eligibility checks and use the regular path when they are not TMA-eligible.

In addition to the environment variable, each CTA that should be eligible for TMA must provide shared memory to NVSHMEM. Query the amount of shared memory with `nvshmemx_ask_smem`, provide at least that amount of CTA shared memory to the kernel, and have one thread in every participating CTA call `nvshmemx_give_smem` once before issuing TMA-eligible RMA operations. Synchronize the CTA after registration before other threads use the TMA path. All CTAs that register shared memory for a kernel launch must provide the same size. Before the kernel returns, one thread in every CTA that called `nvshmemx_give_smem` should call `nvshmemx_release_smem` after the CTA has finished using the TMA path.

The shared memory passed to `nvshmemx_give_smem` may be statically or dynamically allocated CTA shared memory. The region must be at least the size returned by `nvshmemx_ask_smem` and all participating CTAs in a kernel launch must register the same size. Kernels that also use application shared memory can either allocate one shared-memory region and split it into an NVSHMEM prefix and an application region, or pass one shared-memory region to NVSHMEM while keeping unrelated application data in another shared-memory region.

Use `NVSHMEMX_SMEM_RECOMMENDED` for the general case where NVSHMEM may need shared memory for global-memory staging. Use `NVSHMEMX_SMEM_MINIMUM` only when occupancy pressure requires a smaller staging allocation. Use `NVSHMEMX_SMEM_BARRIERS_ONLY` when application data is already in shared memory and NVSHMEM only needs space for its internal TMA state.

    __global__ void put_kernel(float *remote_dst, const float *local_src,
                               size_t nelems, int peer, size_t nvshmem_smem_size) {
        extern __shared__ char nvshmem_smem[];

        if (threadIdx.x == 0) {
            nvshmemx_give_smem(nvshmem_smem, nvshmem_smem_size);
        }
        __syncthreads();

        /*
         * The existing put API is unchanged. With NVSHMEM_TMA_POLICY=ENABLE,
         * SM90+ hardware, a peer-reachable destination, and a compatible
         * transfer shape, this call may use TMA.
         */
        nvshmemx_float_put_nbi_block(remote_dst, local_src, nelems, peer);

        /*
         * quiet provides remote completion and also drains pending TMA work.
         */
        nvshmem_quiet();

        __syncthreads();
        if (threadIdx.x == 0) {
            nvshmemx_release_smem();
        }
    }

    void launch_put(float *remote_dst, const float *local_src,
                    size_t nelems, int peer, cudaStream_t stream) {
        size_t smem_size =
            (size_t)nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);

        if (smem_size > 48 * 1024) {
            cudaFuncSetAttribute(put_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smem_size);
        }

        dim3 grid(1);
        dim3 block(256);
        put_kernel<<<grid, block, smem_size, stream>>>(
            remote_dst, local_src, nelems, peer, smem_size);
    }

When the source data is produced in shared memory, reserve a prefix of the dynamic shared-memory allocation for NVSHMEM and place application data after that region. For this pattern, `nvshmem_smem_size` is typically `nvshmemx_ask_smem(NVSHMEMX_SMEM_BARRIERS_ONLY)`, and the kernel launch should provide enough dynamic shared memory for both the NVSHMEM prefix and the application tile. This lets NVSHMEM use the registered shared memory for internal TMA state while the TMA transfer reads directly from the application’s shared-memory tile.

After writing a shared-memory source tile and before issuing the NVSHMEM put, the kernel must make those shared-memory stores visible to the TMA async proxy engine. The example below uses the SM90+ shared async-proxy fence for that ordering.

    __global__ void put_shared_tile(float *remote_dst, size_t tile_elems,
                                    int peer, size_t nvshmem_smem_size) {
        extern __shared__ char smem[];

        char *nvshmem_smem = smem;
        float *tile = reinterpret_cast<float *>(smem + nvshmem_smem_size);

        if (threadIdx.x == 0) {
            nvshmemx_give_smem(nvshmem_smem, nvshmem_smem_size);
        }
        __syncthreads();

        for (size_t i = threadIdx.x; i < tile_elems; i += blockDim.x) {
            tile[i] = static_cast<float>(i);
        }
        __syncthreads();

    #if __CUDA_ARCH__ >= 900
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    #endif

        nvshmemx_float_put_nbi_block(remote_dst, tile, tile_elems, peer);

        /*
         * quiet waits for remote completion. If only source-tile reuse is
         * needed, nvshmemx_flush_block() is sufficient for the block-scoped put.
         */
        nvshmem_quiet();

        __syncthreads();
        if (threadIdx.x == 0) {
            nvshmemx_release_smem();
        }
    }

For kernels that already use shared memory, the same rule applies: allocate enough shared memory for both NVSHMEM and the application, pass the NVSHMEM portion to `nvshmemx_give_smem`, and keep application-managed shared-memory objects outside the region reserved for NVSHMEM. The NVSHMEM portion may be dynamic or static CTA shared memory. Dynamic shared memory is usually the simplest way to reserve a region, but static shared memory is also supported if the registered region satisfies the size and per-launch consistency requirements above.

## What to Expect

TMA is most useful for large point-to-point transfers over NVLink where the application wants to reduce the number of threads devoted to communication. For puts whose source is in application shared memory, TMA can issue an asynchronous shared-memory to remote global-memory transfer directly from that source operand. For puts whose source and destination operands are both in global memory, NVSHMEM may stage through the shared memory supplied with `nvshmemx_give_smem`. In this release, all TMA-backed gets also use NVSHMEM’s per-CTA internal staging resources.

Nonblocking TMA-backed puts can return before the TMA engine has finished reading the local source buffer. The flush operations make source buffers safe to reuse without waiting for remote visibility for all issuing scopes. Use `nvshmemx_flush` for thread-scoped code, `nvshmemx_flush_warp` for warp-scoped code, and `nvshmemx_flush_block` for block-scoped code. Use `nvshmem_fence` only when subsequent NVSHMEM operations must be ordered after earlier TMA-backed operations; fence does not guarantee remote completion. To make the transfer visible to the remote PE, or when full completion is required, use `nvshmem_quiet`. `nvshmem_quiet` drains pending TMA work before returning.

Put-with-signal operations preserve signal ordering. When the put portion is routed through TMA, NVSHMEM completes the TMA put before issuing the signal update.

## Limitations

TMA support currently has the following limitations:

  * TMA requires SM90 or newer GPUs. With `NVSHMEM_TMA_POLICY=ENABLE`, NVSHMEM disables TMA on older GPUs. With `NVSHMEM_TMA_POLICY=FORCE`, initialization fails on older GPUs.

  * TMA instructions require shared memory as one side of the transfer. For puts whose application source and destination operands are both in global memory, NVSHMEM must stage the data through the shared memory registered with `nvshmemx_give_smem`. A conceptual global-to-global put therefore behaves like:

        local global memory -> CTA shared-memory staging tile -> remote global memory

In this release, all TMA-backed gets use the same per-CTA internal staging resources. A conceptual get behaves like:

        remote global memory -> CTA shared-memory staging tile -> local destination

If a put source operand is already in application shared memory, NVSHMEM can use it directly and avoid the internal staging tile.

  * TMA is used only for peer-reachable GPU memory. Transfers that must use a network transport, such as IB or RoCE, continue to use the regular NVSHMEM transport path. Puts whose source operand is application shared memory require a peer-reachable remote global-memory destination; shared-memory source operands are not supported for network-transport puts.

  * TMA cannot target remote shared memory. The remote operand must be addressable as global memory, such as an NVSHMEM symmetric object on the peer GPU.

  * TMA transfers require 16-byte aligned source and destination addresses and a transfer size that is a multiple of 16 bytes. Transfers that do not satisfy these routing constraints use the regular path.

  * Block-scoped TMA puts from global-memory sources use a double-buffered staging path and require at least two warps in the CTA. Smaller CTAs use the regular path.

  * Each participating CTA must register shared memory with one `nvshmemx_give_smem` call before issuing TMA-eligible operations and may call `nvshmemx_release_smem` before returning from the kernel. CTAs that do not register shared memory are not eligible for the TMA path.

  * TMA shared-memory registration is tracked per CTA for a bounded number of CTAs. The current implementation tracks up to 4096 CTAs. CTAs beyond that limit remain valid but use the regular path.

  * Currently, only one independent entity in a CTA may enter a TMA path that uses NVSHMEM’s per-CTA internal staging resources. This applies to all TMA-backed gets and to TMA-backed puts whose source and destination operands are both in global memory. For example, concurrent thread-scoped or warp-scoped callers in the same CTA can overwrite each other’s staging data or internal mbarrier state. Serialize these staged TMA calls within the CTA. This limitation does not apply to puts whose source operand is application shared memory and whose destination operand is remote global memory, because those puts read directly from the application shared-memory source instead of using NVSHMEM’s internal staging tile.
