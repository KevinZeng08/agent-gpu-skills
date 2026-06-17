# Status and Utility Methods

Methods on [`Communicator`](class.md#nccl.core.Communicator) for resource cleanup and error/status
queries.

## close_all_resources

#### Communicator.close_all_resources() → None

Closes all resources owned by this communicator.

Called automatically during [`destroy()`](lifecycle.md#nccl.core.Communicator.destroy) and [`abort()`](lifecycle.md#nccl.core.Communicator.abort),
but can be called manually. Performs best-effort cleanup, ignoring
any errors that occur during resource deallocation. Idempotent:
safe to call multiple times.

## get_last_error

#### Communicator.get_last_error() → str

Returns the last error string for this communicator.

* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

## get_async_error

#### Communicator.get_async_error() → nccl.bindings.nccl.Result

Queries the progress and potential errors of asynchronous NCCL operations.

Operations without a stream argument (e.g. [`finalize()`](lifecycle.md#nccl.core.Communicator.finalize)) are
complete when they return `ncclSuccess`. Operations with a stream
argument (e.g. [`reduce()`](collectives.md#nccl.core.Communicator.reduce)) return `ncclSuccess` when posted
but may report errors through this method until completed. If any
NCCL function returns `ncclInProgress`, users must query the
communicator state until it becomes `ncclSuccess` before calling
another NCCL function.

Before the state becomes `ncclSuccess`, do not issue CUDA kernels
on streams used by NCCL. If an error occurs, destroy the
communicator with [`abort()`](lifecycle.md#nccl.core.Communicator.abort); nothing can be assumed about
the completion or correctness of enqueued operations after an
error.

* **Returns:**
  Current state of the communicator (`ncclSuccess`,
  `ncclInProgress`, or an error code).
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

#### SEE ALSO
[`ncclCommGetAsyncError()`](../../api/comms.md#c.ncclCommGetAsyncError)

## get_mem_stat

#### Communicator.get_mem_stat(stat: [NcclCommMemStat](#nccl.core.NcclCommMemStat)) → int

Queries communicator memory statistics.

* **Parameters:**
  **stat** – The memory statistic to query.
* **Returns:**
  The memory statistic value (bytes, or 0/1 for GPU_MEM_SUSPENDED).
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

## NcclCommMemStat

### *class* nccl.core.NcclCommMemStat(value, names=<not given>, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: `IntEnum`

Memory-statistic selector, mirroring [`ncclCommMemStat_t`](../../api/types.md#c.ncclCommMemStat_t).

Used as the `stat` argument of [`Communicator.get_mem_stat()`](#nccl.core.Communicator.get_mem_stat)
to identify which memory statistic to query. All values are returned
in bytes except [`GPU_MEM_SUSPENDED`](#nccl.core.NcclCommMemStat.GPU_MEM_SUSPENDED), which is a 0/1 flag.

#### GPU_MEM_SUSPEND *= 0*

Communicator-allocated GPU memory that can be released by
[`Communicator.suspend()`](lifecycle.md#nccl.core.Communicator.suspend) (bytes).

#### GPU_MEM_SUSPENDED *= 1*

Whether communicator-allocated GPU memory is currently suspended
(`0` = active, `1` = suspended).

#### GPU_MEM_PERSIST *= 2*

Communicator-allocated GPU memory that cannot be suspended
(bytes).

#### GPU_MEM_TOTAL *= 3*

Total communicator-allocated GPU memory tracked by NCCL (bytes).

## get_error_string

Module-level helper to render an NCCL result code as a human-readable string.

### nccl.core.get_error_string(nccl_result: \_nccl_bindings.Result | int) → str

Returns a human-readable error string for an NCCL result code.

* **Parameters:**
  **nccl_result** – NCCL result code.
* **Returns:**
  Human-readable error message corresponding to the result code.
