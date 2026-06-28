# Library Handles

The NVSHMEM library provides a set of predefined named constant handles. All named constants can be used in initialization expressions or assignments, but not necessarily in array declarations or as labels in _C_ switch statements. This implies named constants to be link-time but not necessarily compile-time constants.

`NVSHMEM_TEAM_WORLD`

Handle of type `nvshmem_team_t` that corresponds to the world team that contains all PEs in the NVSHMEM program. All point-to-point communication operations and collective synchronizations that do not specify a team are performed on the world team. See Section Team Management for more detail about its use.

`NVSHMEM_TEAM_SHARED`

Handle of type `nvshmem_team_t` that corresponds to a team of PEs that share a memory domain. `NVSHMEM_TEAM_SHARED` refers to the team of all PEs that would mutually return a non-null address from a call to `nvshmem_ptr` for all symmetric heap objects. That is, `nvshmem_ptr` must return a non-null pointer to the local PE for all symmetric heap objects on all target PEs in the team. This means that symmetric heap objects on each PE are directly load/store accessible by all PEs in the team. See Section Team Management for more detail about its use.

`NVSHMEMX_TEAM_NODE`

Handle of type `nvshmem_team_t` that corresponds to the team that contains all PEs in the same node as the calling PE. Using this team in a call to `nvshmem_team_my_pe` may provide an index suitable for use with `cudaSetDevice`.
