# Utility Functions for NVSHMEM4Py

The following are nvshmem.core APIs that can be used as-is from their bindings

`nvshmem.core.direct.``ComparisonType`

alias of `nvshmem.bindings.nvshmem.Cmp_type`

`nvshmem.core.direct.``SignalOp`

alias of `nvshmem.bindings.nvshmem.Signal_op`

`nvshmem.core.direct.``InitStatus`

alias of `nvshmem.bindings.nvshmem.Init_status`

`nvshmem.core.direct. my_pe ( ) → int`

Get the current Processing Element (PE) ID of this process.

Returns:
    int: The PE ID of the calling process within `TEAM_WORLD`.

`nvshmem.core.direct. team_my_pe ( team ) → int`

Get the PE ID of this process within a specified team.

Args:
    team: The team handle (e.g., `nvshmem.core.Teams.TEAM_NODE`).
Returns:
    int: The PE ID of the calling process within the specified team.

`nvshmem.core.direct. team_n_pes ( team ) → int`

Get the number of Processing Elements (PEs) in a specified team.

Args:
    team: The team handle (e.g., `nvshmem.core.Teams.TEAM_NODE`).
Returns:
    int: The total number of PEs in the specified team.

`nvshmem.core.direct. n_pes ( ) → int`

Get the total number of Processing Elements (PEs) in `TEAM_WORLD`.

Returns:
    int: The total number of PEs in the default global team (`TEAM_WORLD`).

`nvshmem.core.direct. init_status ( ) → nvshmem.bindings.nvshmem.Init_status`

Get the current initialization status

Returns:
    InitStatus: An enum representing the status of initialization.
