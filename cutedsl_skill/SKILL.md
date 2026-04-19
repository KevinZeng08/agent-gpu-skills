---
name: cutedsl-skill
description: "CuteDSL kernel development for CUDA. Use when working with CuteDSL, CuTe layouts, TMEM, SMEM, blockscaled GEMM, flash attention kernels, CuTeDSL inline PTX, llvm.inline_asm, NVVM ops, dsl_user_op, tcgen05 instructions, TMEM load/store, SMEM descriptor, tcgen05.mma, tcgen05.mma.ws, store_256b, TMA, mbarrier, or asks about wrapping PTX instructions as Python-callable functions in CuTeDSL."
---

# CuTeDSL Development Guide

## CuTeDSL Source Locations

CuTeDSL is part of the CUTLASS Python package. If using the `cutlass-skill`,
CuTeDSL source is at `cutlass-skill/repos/cutlass/python/CuTeDSL/`.

```
cutlass/python/CuTeDSL/
├── cutlass/
│   ├── base_dsl/       # DSL foundation: types, variables, functions, PTX emit
│   ├── cute/           # CuTe Python bindings: Layout, Tensor, TiledMMA, TiledCopy
│   ├── cutlass_dsl/    # CUTLASS DSL: GEMM builder, epilogue, pipeline
│   ├── pipeline/       # Pipeline abstractions: MainloopPipeline, PipelineAsync
│   ├── utils/          # Compilation tools, profiler, tensor utilities
│   └── torch.py        # PyTorch integration
```

CuTeDSL examples:

```
cutlass/examples/python/CuTeDSL/
├── ampere/             # Ampere: sgemm, tensorop_gemm, flash_attention_v2
├── hopper/             # Hopper: TMA gemm, FP8, grouped GEMM
├── blackwell/          # Blackwell: blockwise_gemm
├── cute/               # CuTe tutorials (Python)
├── notebooks/          # Jupyter notebooks
└── advanced_compiler_control/  # Advanced compilation control
```

## Key Patterns

### Debug Printing
```python
with cute.arch.elect_one():
    cute.printf("value: %d", some_value)
    cute.printf("layout: {}", tensor.layout)  # {} for CuTe types
```

### Tensor Slicing
```python
# None = take all, integer = select index
sliced = tensor[(None, None, idx)]
```

### Loop Constructs
```python
cutlass.range(n, unroll_full=True)  # Required for SSA threading
cutlass.range_constexpr(n)           # Compile-time loop counter (Python int)
```

### Barriers
```python
cute.arch.barrier(barrier_id=id, number_of_threads=count)  # arrive + wait
cute.arch.barrier_arrive(barrier_id=id, number_of_threads=count)  # arrive only
```

## TMA and Synchronization

TMA (Tensor Memory Accelerator, Hopper+) asynchronously copies tiles between GMEM and SMEM via descriptors. It enables single-threaded issuance, automatic OOB predication, and warp specialization. TMA loads use mbarriers for completion tracking; TMA stores require a proxy fence before issuance.

For the complete TMA workflow (descriptor creation, partitioning, pipelines, multicast, store-reduce, and code references), see [TMA Guide](references/tma-guide.md).

## Detailed Guides

- [TMA (Tensor Memory Accelerator) Guide](references/tma-guide.md) — descriptor creation, partitioning flow, synchronization, multicast, store-reduce, code references
- [Debugging SMEM Values](references/debugging-smem-values.md) — printing SMEM as hex, calculating stage strides, diagnosing zero-data issues
- [Inline PTX Integration](references/add-inline-ptx.md) — `@dsl_user_op`, `llvm.inline_asm`, NVVM ops, constraint strings, TMEM/SMEM patterns
