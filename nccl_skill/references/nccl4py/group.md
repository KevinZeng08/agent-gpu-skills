# Group Operations

Free functions and helpers for batching NCCL operations into groups. See
[Group Calls](../usage/groups.md#group-calls) for usage details.

## group

### nccl.core.group() → Generator[None, None, None]

Context manager for NCCL group operations.

Automatically calls [`group_start()`](#nccl.core.group_start) on entry and
[`group_end()`](#nccl.core.group_end) on exit, ensuring proper cleanup even if an
exception occurs.

Simulation mode is not supported here. To simulate, call
[`group_start()`](#nccl.core.group_start) and [`group_end()`](#nccl.core.group_end) directly and pass
`simulate=True` to [`group_end()`](#nccl.core.group_end).

## group_start

### nccl.core.group_start() → None

Starts a group of NCCL operations.

All NCCL operations called after this will be batched together and
executed when [`group_end()`](#nccl.core.group_end) is called. This can improve performance
by allowing NCCL to optimize the operation sequence.

## group_end

### nccl.core.group_end(\*, simulate: Literal[False] = False) → None

### nccl.core.group_end(\*, simulate: Literal[True]) → [GroupSimInfo](#nccl.core.GroupSimInfo)

### nccl.core.group_end(\*, simulate: bool) → [GroupSimInfo](#nccl.core.GroupSimInfo) | None

Ends a group of NCCL operations.

By default, executes all operations queued since the last
[`group_start()`](#nccl.core.group_start). When `simulate=True`, the queued operations
are simulated instead of executed, and the estimated execution time is
returned in a [`GroupSimInfo`](#nccl.core.GroupSimInfo).

* **Parameters:**
  **simulate** – When True, simulates the group instead of executing it
  and returns a [`GroupSimInfo`](#nccl.core.GroupSimInfo) carrying the estimated
  time. Defaults to False.
* **Returns:**
  `None` when `simulate=False`; a [`GroupSimInfo`](#nccl.core.GroupSimInfo) with
  the simulation result when `simulate=True`.

## GroupSimInfo

### *class* nccl.core.GroupSimInfo(estimated_time: float)

Bases: `object`

Result of an NCCL group simulation.

Returned by [`group_end()`](#nccl.core.group_end) when called with `simulate=True`.

#### estimated_time *: float*

Estimated execution time for the simulated group operations, in seconds.
