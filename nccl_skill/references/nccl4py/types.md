# Types and Constants

Type wrappers, predefined value constants, and type aliases that appear
in public method signatures.

## Data type

### NcclDataType

### *class* nccl.core.NcclDataType(value, names=<not given>, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: `IntEnum`

NCCL data type, mirroring [`ncclDataType_t`](../api/types.md#c.ncclDataType_t).

Used as the `dtype` of buffer specs and as the `datatype` argument
of NCCL collective operations. Supports conversion to/from NumPy
dtypes via [`from_numpy_dtype()`](#nccl.core.NcclDataType.from_numpy_dtype) and [`numpy_dtype`](#nccl.core.NcclDataType.numpy_dtype).

#### INT8 *= 0*

Signed 8-bit integer.

#### CHAR *= 0*

Alias of [`INT8`](#nccl.core.NcclDataType.INT8).

#### UINT8 *= 1*

Unsigned 8-bit integer.

#### INT32 *= 2*

Signed 32-bit integer.

#### INT *= 2*

Alias of [`INT32`](#nccl.core.NcclDataType.INT32).

#### UINT32 *= 3*

Unsigned 32-bit integer.

#### INT64 *= 4*

Signed 64-bit integer.

#### UINT64 *= 5*

Unsigned 64-bit integer.

#### FLOAT16 *= 6*

IEEE half-precision floating point (2 bytes).

#### HALF *= 6*

Alias of [`FLOAT16`](#nccl.core.NcclDataType.FLOAT16).

#### FLOAT32 *= 7*

IEEE single-precision floating point (4 bytes).

#### FLOAT *= 7*

Alias of [`FLOAT32`](#nccl.core.NcclDataType.FLOAT32).

#### FLOAT64 *= 8*

IEEE double-precision floating point (8 bytes).

#### DOUBLE *= 8*

Alias of [`FLOAT64`](#nccl.core.NcclDataType.FLOAT64).

#### BFLOAT16 *= 9*

Brain floating-point (16-bit truncated single precision; CUDA 11+).

#### FLOAT8E4M3 *= 10*

8-bit floating point, 4 exponent + 3 mantissa bits (CUDA >= 11.8, SM >= 90).

#### FLOAT8E5M2 *= 11*

8-bit floating point, 5 exponent + 2 mantissa bits (CUDA >= 11.8, SM >= 90).

#### *classmethod* from_numpy_dtype(dtype: numpy.dtype) → [NcclDataType](#nccl.core.NcclDataType)

Maps a NumPy dtype to its NCCL equivalent.

* **Parameters:**
  **dtype** – A NumPy dtype. Mapped first by name (for `ml-dtypes`
  like `bfloat16`, `float8_e4m3fn`, `float8_e5m2`)
  and then by `(kind, itemsize)` for standard types.
* **Returns:**
  Corresponding [`NcclDataType`](#nccl.core.NcclDataType) member.
* **Raises:**
  [**NcclInvalid**](#nccl.core.NcclInvalid) – If the dtype has no NCCL equivalent.

#### *property* itemsize *: int*

Size in bytes of a single element of this data type.

#### *property* numpy_dtype *: numpy.dtype*

Equivalent NumPy dtype.

* **Returns:**
  NumPy dtype corresponding to this NCCL data type. For
  `BFLOAT16` and the float8 variants, `ml-dtypes` must be
  installed.
* **Raises:**
  [**NcclInvalid**](#nccl.core.NcclInvalid) – If `ml-dtypes` is required but not installed.

### Predefined data type constants

Module-level [`NcclDataType`](#nccl.core.NcclDataType) instances for use as the `dtype`
argument of buffer specs.

| Constant               | Maps to                                                         |
|------------------------|-----------------------------------------------------------------|
| `nccl.core.INT8`       | [`NcclDataType.INT8`](#nccl.core.NcclDataType.INT8)             |
| `nccl.core.CHAR`       | [`NcclDataType.CHAR`](#nccl.core.NcclDataType.CHAR)             |
| `nccl.core.UINT8`      | [`NcclDataType.UINT8`](#nccl.core.NcclDataType.UINT8)           |
| `nccl.core.INT32`      | [`NcclDataType.INT32`](#nccl.core.NcclDataType.INT32)           |
| `nccl.core.INT`        | [`NcclDataType.INT`](#nccl.core.NcclDataType.INT)               |
| `nccl.core.UINT32`     | [`NcclDataType.UINT32`](#nccl.core.NcclDataType.UINT32)         |
| `nccl.core.INT64`      | [`NcclDataType.INT64`](#nccl.core.NcclDataType.INT64)           |
| `nccl.core.UINT64`     | [`NcclDataType.UINT64`](#nccl.core.NcclDataType.UINT64)         |
| `nccl.core.FLOAT16`    | [`NcclDataType.FLOAT16`](#nccl.core.NcclDataType.FLOAT16)       |
| `nccl.core.HALF`       | [`NcclDataType.HALF`](#nccl.core.NcclDataType.HALF)             |
| `nccl.core.FLOAT32`    | [`NcclDataType.FLOAT32`](#nccl.core.NcclDataType.FLOAT32)       |
| `nccl.core.FLOAT`      | [`NcclDataType.FLOAT`](#nccl.core.NcclDataType.FLOAT)           |
| `nccl.core.FLOAT64`    | [`NcclDataType.FLOAT64`](#nccl.core.NcclDataType.FLOAT64)       |
| `nccl.core.DOUBLE`     | [`NcclDataType.DOUBLE`](#nccl.core.NcclDataType.DOUBLE)         |
| `nccl.core.BFLOAT16`   | [`NcclDataType.BFLOAT16`](#nccl.core.NcclDataType.BFLOAT16)     |
| `nccl.core.FLOAT8E4M3` | [`NcclDataType.FLOAT8E4M3`](#nccl.core.NcclDataType.FLOAT8E4M3) |
| `nccl.core.FLOAT8E5M2` | [`NcclDataType.FLOAT8E5M2`](#nccl.core.NcclDataType.FLOAT8E5M2) |

## Reduction operator

### NcclRedOp

### *class* nccl.core.NcclRedOp(value, names=<not given>, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: `IntEnum`

NCCL reduction operator, mirroring [`ncclRedOp_t`](../api/types.md#c.ncclRedOp_t).

Used as the `op` argument of reduction collectives
([`Communicator.allreduce()`](communicator/collectives.md#nccl.core.Communicator.allreduce), [`Communicator.reduce()`](communicator/collectives.md#nccl.core.Communicator.reduce),
[`Communicator.reduce_scatter()`](communicator/collectives.md#nccl.core.Communicator.reduce_scatter)).

#### SUM *= 0*

Element-wise sum (`+`).

#### PROD *= 1*

Element-wise product (`*`).

#### MAX *= 2*

Element-wise maximum.

#### MIN *= 3*

Element-wise minimum.

#### AVG *= 4*

Sum across all ranks divided by the number of ranks.

### Predefined reduction operators

Module-level [`NcclRedOp`](#nccl.core.NcclRedOp) instances for use as the `op` argument
of reduction collectives. User-defined operators are created via
[`Communicator.create_pre_mul_sum()`](communicator/collectives.md#nccl.core.Communicator.create_pre_mul_sum).

| Constant         | Maps to                                       |
|------------------|-----------------------------------------------|
| `nccl.core.SUM`  | [`NcclRedOp.SUM`](#nccl.core.NcclRedOp.SUM)   |
| `nccl.core.PROD` | [`NcclRedOp.PROD`](#nccl.core.NcclRedOp.PROD) |
| `nccl.core.MAX`  | [`NcclRedOp.MAX`](#nccl.core.NcclRedOp.MAX)   |
| `nccl.core.MIN`  | [`NcclRedOp.MIN`](#nccl.core.NcclRedOp.MIN)   |
| `nccl.core.AVG`  | [`NcclRedOp.AVG`](#nccl.core.NcclRedOp.AVG)   |

## Exceptions

### NcclInvalid

Python-side validation exception, raised when a public API receives a
malformed argument before it reaches NCCL itself.

### *exception* nccl.core.NcclInvalid(msg)

Bases: `Exception`

Raised when an argument provided to an NCCL4Py API is invalid.

Used for argument validation errors that the Python layer detects before
forwarding the call to NCCL (e.g. unsupported dtype, mismatched buffer
counts, wrong device). Errors raised by NCCL itself are reported as
NCCLError from the bindings layer.
