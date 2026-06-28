# Team Management

The PEs in an NVSHMEM program communicate using either point-to-point routines—such as RMA and AMO routines—that specify the PE number of the target PE, or collective routines that operate over a set of PEs. NVSHMEM teams allow programs to group a set of PEs for communication. Team-based collective operations include all PEs in a valid team. Point-to-point communication can make use of team-relative PE numbering through PE number translation.

## Predefined and Application-Defined Teams

An NVSHMEM team may be predefined (i.e., provided by the NVSHMEM library) or defined by the NVSHMEM application. An application-defined team is created by “splitting” a parent team into one or more new teams—each with some subset of PEs of the parent team—via one of the `nvshmem_team_split_*` routines.

All predefined teams are valid for the duration of the NVSHMEM portion of an application. Any team successfully created by a `nvshmem_team_split_*` routine is valid until it is destroyed. All valid teams have a least one member.

## Team Handles

A _team handle_ is an opaque object with type `nvshmem_team_t` that is used to reference a team. Team handles are not remotely accessible objects. The predefined teams may be accessed via the team handles listed in Section Library Handles.

NVSHMEM communication routines that do not accept a team handle argument operate on the world team, which may be accessed through the `NVSHMEM_TEAM_WORLD` handle. The world team encompasses the set of all PEs in the NVSHMEM program, and a given PE’s number in the world team is equal to the value returned by `nvshmem_my_pe`.

A team handle may be initialized to or assigned the value `NVSHMEM_TEAM_INVALID` to indicate that handle does not reference a valid team. When managed in this way, applications can use an equality comparison to test whether a given team handle references a valid team.

## Thread Safety

When it is allowed by the threading model provided by the NVSHMEM library, a team may be used concurrently in non-collective operations (e.g., `nvshmem_team_my_pe`) by multiple threads within the PE where it was created. A team may not be used concurrently by multiple threads in the same PE for collective operations. However, multiple collective operations on different teams may be performed in parallel.

## Collective Ordering

In NVSHMEM, a team object encapsulates resources used to communicate between PEs in collective operations. When calling multiple subsequent collective operations on a team, the collective operations—along with any relevant team based resources—are matched across the PEs in the team based on ordering of collective routine calls. It is the responsibility of the user to ensure that team-based collectives occur in the same program order across all PEs in a team.

For a full discussion of collective semantics, see Section Collective Communication.

## Team Creation

NVSHMEM provides APIs for creating teams by splitting an existing parent team, or by creating a new team with arbitrary PE alignment and ordering.

### Team Splitting

Team splitting is a collective operation on the parent team object. New teams result from a `nvshmem_team_split_*` routine, which takes a parent team and other arguments and produces new teams that contain a subset of the PEs that are members of the parent

team. All PEs in a parent team must participate in a split operation to create new teams. If a PE from the parent team is not a member of any resulting new teams, it will receive a value of `NVSHMEM_TEAM_INVALID` as the value for the new team handle.

Teams that are created by a `nvshmem_team_split_*` routine may be provided a configuration argument that specifies attributes of each new team. This configuration argument is of type `nvshmem_team_config_t`, which is detailed further in Section NVSHMEM_TEAM_CONFIG_T.

Split operations are collective and are subject to the constraints on team-based collectives specified in Section Collective Communication. In particular, in multithreaded executions, threads at a given PE must not perform simultaneous split operations on the same parent team. Team creation operations are matched across participating PEs based on the order in which they are performed. Thus, team creation events must also occur in the same order on all PEs in the parent team.

### Arbitrary Team Initialization

Team initialization is enabled through the `nvshmemx_team_init` and `nvshmemx_team_get_uniqueid` routines. Team initialization may be performed by any set of PE in the existing `NVSHMEM_TEAM_WORLD` team.

Teams initialized by `nvshmemx_team_init` may contain an arbitrary set of PEs with a non-linear stride and an ordering that is not necessarily increasing with respect to the PE index in `NVSHMEM_TEAM_WORLD`.

Team initialization requires the user to select a base PE to act as the root of the new team. This root will be responsible for acquiring a unique ID and distributing it to all other PEs in the team.

PEs in a newly created team are consecutively numbered starting with PE number 0. PEs are ordered by their PE number in the parent team. Team relative PE numbers can be used for point-to-point operations through using the translation routine `nvshmem_team_translate_pe`.

Upon completion of a team creation operation, the parent and any resulting teams will be immediately usable for any team-based operations, including creating new child teams, without any intervening synchronization.

## **NVSHMEM_TEAM_MY_PE**

`int nvshmem_team_my_pe ( nvshmem_team_t team )`

`__device__ int nvshmem_team_my_pe ( nvshmem_team_t team )`

_team [IN]_
    An NVSHMEM team handle.

**Description**

When `team` specifies a valid team, the `nvshmem_team_my_pe` routine returns the number of the calling PE within the specified team. The number is an integer between \\(0\\) and \\(N-1\\) for a team containing \\(N\\) PEs. Each member of the team has a unique number.

If `team` compares equal to `NVSHMEM_TEAM_INVALID`, then the value `-1` is returned. If `team` is otherwise invalid, the behavior is undefined.

**Returns**

The number of the calling PE within the specified team, or the value `-1` if the team handle compares equal to `NVSHMEM_TEAM_INVALID`.

**Notes**

For the world team, this routine will return the same value as `nvshmem_my_pe`.

## **NVSHMEM_TEAM_N_PES**

`int nvshmem_team_n_pes ( nvshmem_team_t team )`

`__device__ int nvshmem_team_n_pes ( nvshmem_team_t team )`

_team [IN]_
    An NVSHMEM team handle.

**Description**

When `team` specifies a valid team, the `nvshmem_team_n_pes` routine returns the number of PEs in the team. This will always be a value between \\(1\\) and \\(N\\), where \\(N\\) is the total number of PEs running in the NVSHMEM program.

If `team` compares equal to `NVSHMEM_TEAM_INVALID`, then the value `-1` is returned. If `team` is otherwise invalid, the behavior is undefined.

**Returns**

The number of PEs in the specified team, or the value `-1` if the team handle compares equal to `NVSHMEM_TEAM_INVALID`.

**Notes**

For the world team, this routine will return the same value as `nvshmem_n_pes`.

## **NVSHMEM_TEAM_CONFIG_T**

    typedef struct {
      int num_contexts;
    } nvshmem_team_config_t;

None.

**Description**

A team configuration object is provided as an argument to `nvshmem_team_split_*` routines. It specifies the requested capabilities of the team to be created.

The `num_contexts` member is reserved for future use.

When using the configuration structure to create teams, a mask parameter controls which fields may be accessed by the NVSHMEM library. Any configuration parameter value that is not indicated in the mask will be ignored, and the default value will be used instead. Therefore, a program must only set the fields for which it does not want the default value.

A configuration mask is created through a bitwise OR operation of the following library constants. A configuration mask value of `0` indicates that the team should be created with the default values for all configuration parameters.

_None_
    Reserved for future use.

The default values for configuration parameters are:

`num_contexts` = `0`

## **NVSHMEM_TEAM_GET_CONFIG**

`void nvshmem_team_get_config ( nvshmem_team_t team , nvshmem_team_config_t *config )`

_team [IN]_
    An NVSHMEM team handle.
_config [OUT]_
    A pointer to the configuration parameters for the given team.

**Description**

`nvshmem_team_get_config` copies the configuration stored in `team` into the structure pointed to by `config`.

If `team` compares equal to `NVSHMEM_TEAM_INVALID`, then no operation is performed and `config` is not modified. If `team` is otherwise invalid, the behavior is undefined.

**Returns**

None.

## **NVSHMEM_TEAM_TRANSLATE_PE**

`int nvshmem_team_translate_pe ( nvshmem_team_t src_team , int src_pe , nvshmem_team_t dest_team )`

_src_team [IN]_
    An NVSHMEM team handle.
_src_pe [IN]_
    A PE number in `src_team`.
_dest_team [IN]_
    An NVSHMEM team handle.

**Description**

The `nvshmem_team_translate_pe` routine will translate a given PE number in one team into the corresponding PE number in another team. Specifically, given the `src_pe` in `src_team`, this routine returns that PE’s number in `dest_team`. If `src_pe` is not a member of both `src_team` and `dest_team`, a value of `-1` is returned.

If at least one of `src_team` and `dest_team` compares equal to `NVSHMEM_TEAM_INVALID`, then `-1` is returned. If either of the `src_team` or `dest_team` handles are otherwise invalid, the behavior is undefined.

**Returns**

The specified PE’s number in the `dest_team`, or a value of `-1` if any team handle arguments are invalid or the `src_pe` is not in both the source and destination teams.

**Notes**

If `NVSHMEM_TEAM_WORLD` is provided as the `dest_team` parameter, this routine acts as a global PE number translator and will return the corresponding `NVSHMEM_TEAM_WORLD` number.

The following example demonstrates the use of the team PE number translation routine. The program makes a new team of all of the even number PEs in the world team. Then, all PEs in the new team acquire their PE number in the new team and translate it to the PE number in the world team. ./example_code/shmem_team_translate_pe.c

## **NVSHMEM_TEAM_SPLIT_STRIDED**

`int nvshmem_team_split_strided ( nvshmem_team_t parent_team , int start , int stride , int size , const nvshmem_team_config_t *config , long config_mask , nvshmem_team_t *new_team )`

_parent_team [IN]_
    An NVSHMEM team.
_start [IN]_
    The lowest PE number of the subset of PEs from the parent team that will form the new team.
_stride [IN]_
    The stride between team PE numbers in the parent team that comprise the subset of PEs that will form the new team.
_size [IN]_
    The number of PEs from the parent team in the subset of PEs that will form the new team. `size` must be a positive integer.
_config [IN]_
    A pointer to the configuration parameters for the new team.
_config_mask [IN]_
    The bitwise mask representing the set of configuration parameters to use from `config`.
_new_team [OUT]_
    An NVSHMEM team handle. Upon successful creation, it references an NVSHMEM team that contains the subset of all PEs in the parent team specified by the PE triplet provided.

**Description**

The `nvshmem_team_split_strided` routine is a collective routine. It creates a new NVSHMEM team from an existing parent team, where the PE subset of the resulting team is defined by the triplet of arguments (`start`, `stride`, `size`). A valid triplet is one such that:

\\[start + stride \cdot i \in \mathbb{Z}_{N-1} \; \forall \; i \in \mathbb{Z}_{size-1}\\]

where \\(\mathbb{Z}\\) is the set of natural numbers (\\(0, 1, \dots\\)), \\(N\\) is the number of PEs in the parent team and \\(size\\) is a positive number indicating the number of PEs in the new team. The index \\(i\\) specifies the number of the given PE in the new team. Thus, PEs in the new team remain in the same relative order as in the parent team.

This routine must be called by all PEs in the parent team. All PEs must provide the same values for the PE triplet. On successful creation of the new team:

  * The `new_team` handle will reference a valid team for the subset of PEs in the parent team that are members of the new team.
  * Those PEs in the parent team that are not members of the new team will have `new_team` assigned to `NVSHMEM_TEAM_INVALID`.
  * `nvshmem_team_split_strided` will return zero to all PEs in the parent team.

If the new team cannot be created or an invalid PE triplet is provided, then `new_team` will be assigned the value `NVSHMEM_TEAM_INVALID` and `nvshmem_team_split_strided` will return a nonzero value on all PEs in the parent team.

The `config` argument specifies team configuration parameters, which are described in Section NVSHMEM_TEAM_CONFIG_T.

The `config_mask` argument is a bitwise mask representing the set of configuration parameters to use from `config`. A `config_mask` value of `0` indicates that the team should be created with the default values for all configuration parameters. See Section NVSHMEM_TEAM_CONFIG_T for field mask names and default configuration parameters.

If `parent_team` compares equal to `NVSHMEM_TEAM_INVALID`, then no new team will be created and `new_team` will be assigned the value `NVSHMEM_TEAM_INVALID`. If `parent_team` is otherwise invalid, the behavior is undefined.

**Returns**

Zero on successful creation of `new_team`; otherwise, nonzero.

**Notes**

The `nvshmem_team_split_strided` operation uses an arbitrary `stride` argument, whereas the `logPE_stride` argument to the active set collective operations only permits strides that are a power of two. Arbitrary strides allow a greater number of PE subsets to be expressed and can support a broader range of usage models.

See the description of team handles and predefined teams in Section Team Management for more information about team handle semantics and usage.

The following example demonstrates the use of strided split in a _C_ program. The program creates a new team of all even number PEs from the world team, then retrieves the PE number and team size on all PEs that are members of the new team. ./example_code/shmem_team_split_strided.c

## **NVSHMEM_TEAM_SPLIT_2D**

`int nvshmem_team_split_2d ( nvshmem_team_t parent_team , int xrange , const nvshmem_team_config_t *xaxis_config , long xaxis_mask , nvshmem_team_t *xaxis_team , const nvshmem_team_config_t *yaxis_config , long yaxis_mask , nvshmem_team_t *yaxis_team )`

_parent_team [IN]_
    A valid NVSHMEM team. Any predefined teams, such as `NVSHMEM_TEAM_WORLD`, may be used, or any team created by the user.
_xrange [IN]_
    A positive integer representing the number of elements in the first dimension.
_xaxis_config [IN]_
    A pointer to the configuration parameters for the new `x`-axis team.
_xaxis_mask [IN]_
    The bitwise mask representing the set of configuration parameters to use from `xaxis_config`.
_xaxis_team [OUT]_
    A new PE team handle representing a PE subset consisting of all the PEs that have the same coordinate along the `y`-axis as the calling PE.
_yaxis_config [IN]_
    A pointer to the configuration parameters for the new `y`-axis team.
_yaxis_mask [IN]_
    The bitwise mask representing the set of configuration parameters to use from `yaxis_config`.
_yaxis_team [OUT]_
    A new PE team handle representing a PE subset consisting of all the PEs that have the same coordinate along the `x`-axis as the calling PE.

**Description**

The `nvshmem_team_split_2d` routine is a collective operation. It returns two new teams to the calling PE by splitting an existing parent team into subsets based on a 2D Cartesian space. The user provides the size of the `x` dimension, which is then used to derive the size of the `y` dimension based on the size of the parent team. The size of the `y` dimension will be equal to \\(\lceil N \div xrange \rceil\\), where `N` is the size of the parent team. In other words, \\(xrange \times yrange \geq N\\), so that every PE in the parent team has a unique `(x,y)` location in the 2D Cartesian space. The resulting `xaxis_team` and `yaxis_team` correspond to the calling PE’s row and column, respectively, in the 2D Cartesian space.

The mapping of PE number to coordinates is \\((x, y) = ( pe \mod xrange, \lfloor pe \div xrange \rfloor )\\), where \\(pe\\) is the PE number in the parent team. For example, if \\(xrange = 3\\), then the first 3 PEs in the parent team will form the first `xteam`, the second three PEs in the parent team form the second `xteam`, and so on.

Thus, after the split operation, each of the new `xteam` s will contain all PEs that have the same coordinate along the `y`-axis as the calling PE. Each of the new `yteam` s will contain all PEs with the same coordinate along the `x`-axis as the calling PE.

The PEs are numbered in the new teams based on the coordinate of the PE along the given axis. As a result, the value returned by `nvshmem_team_my_pe(` `xteam``)` is the `x`-coordinate and the value returned by `nvshmem_team_my_pe(` `yteam``)` is the `y`-coordinate of the calling PE.

Any valid NVSHMEM team can be used as the parent team. This routine must be called by all PEs in the parent team. The value of `xrange` must be positive and all PEs in the parent team must pass the same value for `xrange`. When `xrange` is greater than the size of the parent team, `nvshmem_team_split_2d` behaves as though `xrange` were equal to the size of the parent team.

The `xaxis_config` and `yaxis_config` arguments specify team configuration parameters for the `x`\- and `y`-axis teams, respectively. These parameters are described in Section NVSHMEM_TEAM_CONFIG_T. All PEs that will be in the same resultant team must specify the same configuration parameters. The PEs in the parent team _do not_ have to all provide the same parameters for new teams.

The `xaxis_mask` and `yaxis_mask` arguments are a bitwise masks representing the set of configuration parameters to use from `xaxis_config` and `yaxis_config`, respectively. A mask value of `0` indicates that the team should be created with the default values for all configuration parameters. See Section NVSHMEM_TEAM_CONFIG_T for field mask names and default configuration parameters.

If `parent_team` compares equal to `NVSHMEM_TEAM_INVALID`, then no new teams will be created and both `xaxis_team` and `yaxis_team` will be assigned the value `NVSHMEM_TEAM_INVALID`. If `parent_team` is otherwise invalid, the behavior is undefined.

If any `xaxis_team` or `yaxis_team` on any PE in `parent_team` cannot be created, then both team handles on all PEs in `parent_team` will be assigned the value `NVSHMEM_TEAM_INVALID` and `nvshmem_team_split_2d` will return a nonzero value.

**Returns**

Zero on successful creation of all `xaxis_team` s and `yaxis_team` s; otherwise, nonzero.

**Notes**

Since the split may result in a 2D space with more points than there are members of the parent team, there may be a final, incomplete row of the 2D mapping of the parent team. This means that the resultant `yteam` s may vary in size by up to 1 PE, and that there may be one resultant `xteam` of smaller size than all of the other `xteam` s.

The following grid shows the 12 teams that would result from splitting a parent team of size 10 with `xrange` of 3. The numbers in the grid cells are the PE numbers in the parent team. The rows are the `xteam` s. The columns are the `yteam`s.

xteam, y=0 | 0 | 1 | 2
---|---|---|---
xteam, y=1 | 3 | 4 | 5
xteam, y=2 | 6 | 7 | 8
xteam, y=3 | 9 |  |
0-1 |  |  |

It would be legal, for example, if PEs 0, 3, 6, 9 specified a different value for `yaxis_config` than all of the other PEs, as long as the configuration parameters match for all PEs in each of the new teams.

See the description of team handles and predefined teams in Section Team Management for more information about team handle semantics and usage.

The following example demonstrates the use of 2D Cartesian split in a _C/C++_ program. This example shows how multiple 2D splits can be used to generate a 3D Cartesian split. ./example_code/shmem_team_split_2D.c

The example above splits `NVSHMEM_TEAM_WORLD` into a 3D team with dimensions `xdim`, `ydim`, and `zdim`, where each dimension is calculated using the functions, `find_xy_dims` and `find_xyz_dims`. When running with 12 PEs, the dimensions are 3x2x2, respectively, and the first split of `NVSHMEM_TEAM_WORLD` results in 4 `xteams` and 3 `yzteams`:

The second split of `yzteam` for `x` = 0, `ydim` = 2 results in 2 `yteams` and 2 `zteams`:

The second split of `yzteam` for `x` = 1, `ydim` = 2 results in 2 `yteams` and 2 `zteams`:

The second split of `yzteam` for `x` = 2, `ydim` = 2 results in 2 `yteams` and 2 `zteams`:

The final number of teams for each dimension are:

  * 4 `xteams`: these are teams where (`z`, `y`) is fixed and `x` varies.
  * 6 `yteams`: these are teams where (`x`, `z`) is fixed and `y` varies.
  * 6 `zteams`: these are teams where (`x`, `y`) is fixed and `z` varies.

The expected output with 12 PEs is:

`xdim = 3, ydim = 2, zdim = 2`

`(0, 0, 0) is mype = 0`

`(1, 0, 0) is mype = 1`

`(2, 0, 0) is mype = 2`

`(0, 1, 0) is mype = 3`

`(1, 1, 0) is mype = 4`

`(2, 1, 0) is mype = 5`

`(0, 0, 1) is mype = 6`

`(1, 0, 1) is mype = 7`

`(2, 0, 1) is mype = 8`

`(0, 1, 1) is mype = 9`

`(1, 1, 1) is mype = 10`

`(2, 1, 1) is mype = 11`

## **NVSHMEM_TEAM_DESTROY**

`void nvshmem_team_destroy ( nvshmem_team_t team )`

_team [IN]_
    An NVSHMEM team handle.

**Description**

The `nvshmem_team_destroy` routine is a collective operation that destroys the team referenced by the team handle argument `team`. Upon return, the referenced team is invalid.

If `team` compares equal to `NVSHMEM_TEAM_WORLD` or any other predefined team, the behavior is undefined.

If `team` compares equal to `NVSHMEM_TEAM_INVALID`, then no operation is performed. If `team` is otherwise invalid, the behavior is undefined.

**Returns**

None.

## **NVSHMEMX_TEAM_INIT**

`int nvshmemx_team_init ( nvshmem_team_t *team , nvshmem_team_config_t *config , long config_mask , int npes , int pe_idx_in_team )`

_team [OUT]_
    A pointer to a `nvshmem_team_t` handle that will be initialized to the new team handle.
_config [IN]_
    A pointer to a `nvshmem_team_config_t` structure that specifies the configuration of the new team.
_config_mask [IN]_
    A bitmask that specifies which fields of the `nvshmem_team_config_t` structure are valid.
_npes [IN]_
    The number of PEs in the new team. Must match for all PEs in the new team.
_pe_idx_in_team [IN]_
    The index of the PE in the new team.

**Description**

This routine initializes a new NVSHMEM team of arbitrary size and ordering. The new team will contain _npes_ PEs and it will be indexed based on the _pe_idx_in_team_ argument.

**Returns**

Returns 0 on success and non-zero on failure.

**Notes**

The following requirements must be met, or the API will return an error:

  * The _config_ argument’s uniqueid field must be set to a valid value supplied by the `nvshmemx_team_get_uniqueid()` routine.
  * The `NVSHMEM_TEAM_CONFIG_MASK_UNIQUEID` bit must be set in the _config_mask_ argument to indicate that the _config_ argument is valid.

Failing to meet the following requirements will result in undefined behavior:

  * The _npes_ argument must match for all PEs in the new team.
  * The _pe_idx_in_team_ argument must be unique for each PE in the new team.
  * The _pe_idx_in_team_ argument must be between 0 and _npes_ \- 1.

## **NVSHMEMX_TEAM_GET_UNIQUEID**

`int nvshmemx_team_get_uniqueid ( nvshmemx_team_uniqueid_t *uniqueid )`

_uniqueid [OUT]_
    A pointer to a `nvshmemx_team_uniqueid_t` handle that will be initialized to the new unique ID.

**Description**

This routine returns a unique ID for the team in the _uniqueid_ argument.

**Returns**

Returns 0 on success and non-zero on failure.

**Notes**

This function may not be used interchangeably with `nvshmemx_get_uniqueid` function which is provided for use with the NVSHMEM UID bootstrap module. Trying to use a uniqueid supplied from that API to initialize a team will result in undefined behavior.
