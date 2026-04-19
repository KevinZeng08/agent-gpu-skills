---
name: cutedsl-add-inline-ptx
description: "Wrap low-level PTX instructions as Python-callable CuTeDSL functions via llvm.inline_asm or native NVVM MLIR ops. Use when adding new PTX wrappers in CuTeDSL, working with @dsl_user_op, tcgen05.ld/st, tcgen05.mma.ws, st.global.L1::no_allocate, TMEM address space, SMEM descriptors, or debugging CuTeDSL inline assembly issues ($N placeholders, constraint strings, !llvm.struct extractvalue, has_side_effects, range_constexpr)."
---

# CuteDSL Inline PTX Integration

## Overview

This skill describes how to wrap low-level PTX instructions as Python-callable
functions in CuteDSL (CUTLASS Python DSL). There are two integration paths:

1. **`llvm.inline_asm`** — for PTX instructions not exposed by NVVM (e.g.
   `tcgen05.mma.ws`, `tcgen05.ld/st`, `st.global.L1::no_allocate`).
2. **Native NVVM ops** — for instructions already modeled in the `nvvm` MLIR
   dialect (e.g. `_nvvm.tcgen05_mma` with `write_disable_mask`).

Both paths use the `@dsl_user_op` decorator to drop into raw MLIR IR building
inside an otherwise high-level CuteDSL kernel.

---

## Imports

```python
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass._mlir import ir as _ir_mod          # or: from cutlass._mlir import ir
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.typing import Int32             # typed wrappers
```

---

## Core Pattern: `@dsl_user_op` Inside `@cute.jit`

The fundamental pattern is a **two-layer structure**: an outer `@cute.jit`
function that handles CuteDSL-level type wrapping, and an inner `@dsl_user_op`
function that builds raw MLIR IR.

```python
@cute.jit
def my_ptx_wrapper(compile_time_param: int, runtime_arg: int):
    """Public CuteDSL API — called from @cute.kernel or other @cute.jit."""

    # Pre-compute ASM string using compile-time params (pure Python)
    asm_str = f"some.ptx.instruction ${0}, ${1};"
    constraints = "=f,r"

    @dsl_user_op
    def _do(arg_val, *, loc=None, ip=None):
        # Build MLIR IR here — full access to llvm/arith/nvvm dialects
        operands = [_to_ir(arg_val, loc, ip)]
        result = llvm.inline_asm(
            result_type,        # or None for void
            operands,
            asm_str,
            constraints,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc, ip=ip,
        )
        return result

    # Call _do with CuteDSL-typed values
    return _do(Int32(runtime_arg))
```

### The `_to_ir()` Helper

Every `@dsl_user_op` file should define this helper to extract raw MLIR values:

```python
def _to_ir(val, loc=None, ip=None):
    """Extract raw MLIR IR value from a CuteDSL wrapper."""
    return val.ir_value(loc=loc, ip=ip) if hasattr(val, "ir_value") else val
```

---

## Critical Rules

### 1. ASM Placeholder Syntax: `$N` Not `%N`

LLVM inline assembly uses **`$0`, `$1`, `$2`, ...** for operand placeholders.
PTX `%` syntax is **wrong** here and will cause silent miscompilation.

```python
# CORRECT
asm_str = "tcgen05.ld.sync.aligned.32x32b.x4.b32 {$0, $1, $2, $3}, [$4];"

# WRONG — will fail or produce garbage
asm_str = "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
```

### 2. Constraint String Format

Constraints follow LLVM inline asm convention. Common constraints:

| Constraint | Meaning | CuteDSL Type |
|-----------|---------|--------------|
| `"=f"` | Output: 32-bit float register | `F32Type` |
| `"f"` | Input: 32-bit float register | `Float32` |
| `"r"` | Input: 32-bit integer register | `Int32` |
| `"l"` | Input: 64-bit integer/pointer | `Int64` / `.toint()` |

- Output constraints come first, separated by commas
- `"=f,=f,r"` means: 2 float outputs, 1 int input
- For multiple outputs, use `!llvm.struct<(f32, f32, ...)>` as `result_type`

### 3. No `raise` Inside `@cute.jit`

CuteDSL's AST preprocessor forbids `raise` statements inside `@cute.jit`
or `@dsl_user_op`. Move validation logic **outside** the decorated function:

```python
# CORRECT — validation in plain Python scope
def tcgen05_ld_32x32b(num: int, taddr: int):
    assert num in (1, 2, 4, 8, 16, 32, 64, 128), f"Invalid num={num}"

    @cute.jit
    def _impl(taddr: int):
        @dsl_user_op
        def _do(addr_val, *, loc=None, ip=None):
            ...
        return _do(Int32(taddr))
    return _impl(taddr)
```

Or simply use `assert` / validation before the `@cute.jit` layer.

### 4. Pointer Handling — `.toint()` for `"l"` Constraint

CuTe iterators (pointers) cannot be passed directly as `"l"` (64-bit)
operands. Call `.toint()` to obtain a raw `i64` value:

```python
# For global memory pointers with "l" constraint:
gmem_addr = my_tensor.iterator.toint()   # → i64 raw address

# Byte-offset arithmetic (not element offset!):
addr_with_offset = gmem_addr + chunk_idx * 32  # 32 bytes = 8 × f32
```

### 5. Multi-Return: `!llvm.struct` + `extractvalue`

When a PTX instruction returns multiple registers (e.g. `tcgen05.ld`),
use an LLVM struct as the result type and extract each field:

```python
@dsl_user_op
def _do(addr_val, *, loc=None, ip=None):
    f32_ty = _ir_mod.F32Type.get()
    num = 4
    res_ty = _ir_mod.Type.parse(
        f"!llvm.struct<({', '.join(['f32'] * num)})>"
    )
    result = llvm.inline_asm(res_ty, [...], asm_str, "=f,=f,=f,=f,r", ...)
    return [
        llvm.extractvalue(f32_ty, result, [i], loc=loc, ip=ip)
        for i in range(num)
    ]
```

### 6. Void Return

For instructions with no output (stores, MMA), pass `None` as `result_type`:

```python
llvm.inline_asm(
    None,                          # ← void return
    operands,
    asm_str,
    "r,l,l,r,r",                  # all input constraints, no "="
    has_side_effects=True,         # MUST be True for stores/MMA
    ...
)
```

### 7. `has_side_effects=True`

Always set `has_side_effects=True` for:
- Store instructions (TMEM, global, shared)
- MMA instructions (write to TMEM accumulators)
- Any instruction with observable side effects

The compiler may reorder or eliminate the instruction without this flag.

### 8. Use `range_constexpr` for Compile-Time Loops

Inside `@cute.jit`, `cutlass.range()` generates MLIR loop IR — the loop
variable is an MLIR value, not a Python int. For indexing Python lists or
generating unrolled code, use `cutlass.range_constexpr()`:

```python
# CORRECT — compile-time unrolling, idx is a Python int
for idx in cutlass.range_constexpr(4):
    store_256b(base_addr + idx * 32, values[idx*8 : (idx+1)*8])

# WRONG — idx is an MLIR value, cannot index Python lists
for idx in cutlass.range(4):
    store_256b(base_addr + idx * 32, values[idx*8 : (idx+1)*8])
```

---

## Pattern A: `llvm.inline_asm` (PTX Not in NVVM)

Use this when the PTX instruction has no corresponding NVVM MLIR op.
This is the most common path for new/experimental instructions.

### Example: `tcgen05.mma.ws` (Weight-Stationary MMA)

```python
@cute.jit
def tcgen05mma_ws_ss_tf32(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    asm_str = (
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, $4, 0;\n"
        "tcgen05.mma.ws.cta_group::1.kind::tf32 "
        "[$0], $1, $2, $3, p;\n"
        "}"
    )

    @dsl_user_op
    def _do(c_val, da_val, db_val, dv_val, sc_val, *, loc=None, ip=None):
        llvm.inline_asm(
            None,                                  # void return
            [
                _ir(c_val,  loc, ip),              # $0: tmem_c (r)
                _ir(da_val, loc, ip),              # $1: desc_a (l = i64)
                _ir(db_val, loc, ip),              # $2: desc_b (l = i64)
                _ir(dv_val, loc, ip),              # $3: desc_val (r)
                _ir(sc_val, loc, ip),              # $4: scale_out (r)
            ],
            asm_str,
            "r,l,l,r,r",                          # constraints match operands
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc, ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        desc_a.desc_i64[0],                       # i64 SMEM descriptor
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
    )
```

**Key points:**
- Predicate registers (`.reg .pred p`) must be declared in a `{ }` block
- TMEM addresses use `[$N]` (memory operand syntax) with `"r"` constraint
- SMEM descriptors are 64-bit → use `"l"` constraint and pass `.desc_i64[0]`

### Example: `st.global.L1::no_allocate.v8.f32` (256-bit Direct R2G Store)

```python
_STORE_256B_ASM = (
    "st.global.L1::no_allocate.v8.f32 [$0], "
    "{$1, $2, $3, $4, $5, $6, $7, $8};"
)
_STORE_256B_CONSTRAINTS = "l,f,f,f,f,f,f,f,f"

@cute.jit
def store_256b(gmem_ptr, values):
    @dsl_user_op
    def _do(addr, s0, s1, s2, s3, s4, s5, s6, s7, *, loc=None, ip=None):
        operands = [
            _to_ir(addr, loc, ip),
            _to_ir(s0, loc, ip), _to_ir(s1, loc, ip),
            _to_ir(s2, loc, ip), _to_ir(s3, loc, ip),
            _to_ir(s4, loc, ip), _to_ir(s5, loc, ip),
            _to_ir(s6, loc, ip), _to_ir(s7, loc, ip),
        ]
        llvm.inline_asm(
            _ir_mod.Type.parse("!llvm.void"),
            operands,
            _STORE_256B_ASM,
            _STORE_256B_CONSTRAINTS,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc, ip=ip,
        )

    _do(gmem_ptr, values[0], values[1], values[2], values[3],
        values[4], values[5], values[6], values[7])
```

**Key points:**
- Global memory pointer uses `"l"` constraint (64-bit address)
- Pass `.toint()` result or raw i64 as the address
- `_ir_mod.Type.parse("!llvm.void")` and `None` are both valid for void return

---

## Pattern B: Native NVVM Ops (Instruction in MLIR Dialect)

When the instruction already exists in the `nvvm` MLIR dialect, prefer
using the native op — it gives the compiler better optimization visibility.

### Example: `tcgen05.mma` with `write_disable_mask`

```python
@dsl_user_op
def _do(c_val, da_val, db_val, dv_val, sc_val,
        m0_val, m1_val, m2_val, m3_val, *, loc=None, ip=None):
    ptr6_ty  = llvm.PointerType.get(address_space=6)     # TMEM address space
    i32_ty   = _ir_mod.IntegerType.get_signless(32)
    i1_ty    = _ir_mod.IntegerType.get_signless(1)
    vec4i32  = _ir_mod.VectorType.get([4], i32_ty)

    # Convert TMEM address int → address_space(6) pointer
    c_ir = _ir(c_val, loc, ip)
    d_ptr = llvm.inttoptr(ptr6_ty, c_ir, loc=loc, ip=ip)

    # Build vector<4xi32> mask from 4 scalar i32 values
    undef = llvm.mlir_undef(vec4i32, loc=loc, ip=ip)
    idx = [_arith.constant(i32_ty, i, loc=loc, ip=ip) for i in range(4)]
    v = llvm.InsertElementOp(undef, _ir(m0_val, loc, ip), idx[0], loc=loc, ip=ip)
    v = llvm.InsertElementOp(v,     _ir(m1_val, loc, ip), idx[1], loc=loc, ip=ip)
    v = llvm.InsertElementOp(v,     _ir(m2_val, loc, ip), idx[2], loc=loc, ip=ip)
    mask = llvm.InsertElementOp(v,  _ir(m3_val, loc, ip), idx[3], loc=loc, ip=ip)

    # Truncate scale_out (i32) → i1 predicate
    enable_d = _arith.trunci(i1_ty, _ir(sc_val, loc, ip), loc=loc, ip=ip)

    # Issue the native NVVM op
    _nvvm.tcgen05_mma(
        mma_kind=_nvvm.Tcgen05MMAKind.TF32,
        cta_group=_nvvm.Tcgen05GroupKind.CTA_1,
        d=d_ptr,
        a=_ir(da_val, loc, ip),       # i64 SMEM descriptor
        b=_ir(db_val, loc, ip),
        idesc=_ir(dv_val, loc, ip),
        enable_input_d=enable_d,       # i1 predicate
        write_disable_mask=mask,       # vector<4xi32>
        loc=loc, ip=ip,
    )
```

**NVVM-specific idioms:**
- `llvm.PointerType.get(address_space=6)` — TMEM pointer space on Blackwell
- `llvm.inttoptr` to convert i32 TMEM address → typed pointer
- `llvm.mlir_undef` + `llvm.InsertElementOp` to build MLIR vectors
- `_arith.trunci(i1_ty, ...)` to convert i32 → i1 predicate
- `_arith.constant(i32_ty, value)` for literal MLIR constants

---

## SMEM Descriptor Construction

For `tcgen05.mma` instructions, SMEM operands require a 64-bit descriptor
encoding base address, leading/stride byte offsets, and swizzle mode.

```python
class Tcgen05SmemDescriptor:
    def __init__(self, desc_64=None):
        self.desc    = cute.make_rmem_tensor((2,), dtype=cutlass.Int32)
        self.desc_i64 = cute.make_tensor(
            cute.recast_ptr(self.desc.iterator, dtype=cute.Int64), (1,)
        )
        if desc_64 is not None:
            self.desc_i64[0] = desc_64

    def __add__(self, byte_offset):
        """Return a new descriptor offset by byte_offset bytes."""
        res     = cute.make_rmem_tensor((2,), dtype=cutlass.Int32)
        res_i64 = cute.make_tensor(
            cute.recast_ptr(res.iterator, dtype=cute.Int64), (1,)
        )
        res[0] = self.desc[0] + (byte_offset >> 4)  # adjust start_address
        res[1] = self.desc[1]                        # high word unchanged
        return Tcgen05SmemDescriptor(res_i64[0])
```

Key technique: store as 2×i32 for bitfield manipulation, alias as 1×i64 via
`cute.recast_ptr` for passing to PTX `"l"` constraint operands.

---

## Common Pitfalls & Solutions

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| `%N` in asm string | Silent wrong results or LLVM error | Use `$N` |
| `raise` in `@cute.jit` | `DSLAstPreprocessorError` | Move validation outside decorator |
| `cutlass.range()` for list indexing | Type error (MLIR value as index) | Use `cutlass.range_constexpr()` |
| Iterator as `"l"` operand | Type mismatch | Call `.toint()` first |
| Missing `has_side_effects=True` | Instruction optimized away | Always set for stores/MMA |
| Element offset for `store_256b` | Wrong memory location | Use **byte** offset: `n_elems * sizeof(dtype)` |
| Duplicate `@cute.jit` + `@dsl_user_op` | Nesting error | Only one decorator per function level |

---

## Reference Files

- **`cula/ops/intrinsics_sm100.py`** — Canonical examples of `llvm.inline_asm`
  pattern: `tcgen05_ld_32x32b`, `tcgen05_st_32x32b`, `store_256b`
- **`cula/ops/ptx_umma_ext.py`** — Both NVVM native ops (masked MMA) and
  `llvm.inline_asm` (WS MMA), plus `Tcgen05SmemDescriptor` class
- **`tests/test_ptx_umma_ws.py`** — End-to-end test kernels showing T2R
  (`tcgen05_ld_32x32b`) → R2G (`store_256b`) data path with `.toint()` and
  byte-offset arithmetic

---

## Step-by-Step: Adding a New PTX Wrapper

1. **Identify the PTX instruction** — check the PTX ISA reference for operand
   types, register counts, and any predicate operands.

2. **Check NVVM dialect** — search `from cutlass._mlir.dialects import nvvm`
   for an existing op. If found, prefer Pattern B (native NVVM).

3. **Write the ASM string** — use `$N` placeholders, declare `.reg .pred` in
   a `{ }` block if needed.

4. **Build the constraint string** — `"="` prefix for outputs, match each `$N`
   to a constraint character (`r`=i32, `l`=i64, `f`=f32).

5. **Choose the return type**:
   - Void: `None` or `_ir_mod.Type.parse("!llvm.void")`
   - Single f32: `_ir_mod.F32Type.get()`
   - Multiple: `!llvm.struct<(f32, f32, ...)>` + `llvm.extractvalue`

6. **Wrap in `@cute.jit` + `@dsl_user_op`** — outer function takes CuteDSL
   types, inner function builds MLIR IR.

7. **Type-wrap arguments** at the call site:
   - `cutlass.Int32(val)` for `"r"` constraint
   - `desc.desc_i64[0]` or `ptr.toint()` for `"l"` constraint
   - Raw float values for `"f"` constraint

8. **Test** — write a `@cute.kernel` that calls the wrapper, compare output
   against a reference (e.g. PyTorch or NumPy).
