# Initialization and Finalization

This section documents the initialization and finalization APIs in `nvshmem.core.init_fini`.

The nvshmem4py.core module provides initialization and finalization routines for the NVSHMEM runtime. These must be called before and after using any NVSHMEM features in Python.

## Examples

**MPI-based initialization:**

    from mpi4py import MPI
    from cuda.core.experimental import Device
    import nvshmem.core as nvshmem

    rank = MPI.COMM_WORLD.Get_rank()
    dev = Device(rank % system.num_devices)
    dev.set_current()

    nvshmem.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")

    # ... use NVSHMEM ...

    nvshmem.finalize()

**UID-based initialization:**

    from mpi4py import MPI
    from cuda.core.experimental import Device, system
    import nvshmem.core as nvshmem
    import numpy as np

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    dev = Device(rank % system.num_devices)
    dev.set_current()

    uid = nvshmem.get_unique_id(empty=(rank != 0))
    comm.Bcast(uid._data.view(np.int8), root=0)

    nvshmem.init(device=dev, uid=uid, rank=rank, nranks=nranks, initializer_method="uid")

    # ... use NVSHMEM ...

    nvshmem.finalize()

**Emulated MPI initialization:**

    from mpi4py import MPI
    from cuda.core.experimental import Device
    import nvshmem.core as nvshmem

    rank = MPI.COMM_WORLD.Get_rank()
    dev = Device(rank % system.num_devices)
    dev.set_current()

    nvshmem.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="emulated_mpi")

    # ... use NVSHMEM ...

    nvshmem.finalize()

## API Reference

## Teams

`class nvshmem.core. Teams ( IntEnum )`

NVSHMEM4Py uses an enumerator to refer to NVSHMEM Teams.

> `TEAM_WORLD`
>
>
> The world team that contains all PEs in the NVSHMEM program.
>
> `TEAM_SHARED`
>
>
> The team of PEs that share a memory domain. `NVSHMEM_TEAM_SHARED` refers to the team of all PEs that would mutually return a non-null address from a call to nvshmem_ptr for all symmetric heap objects. That is, nvshmem_ptr must return a non-null pointer to the local PE for all symmetric heap objects on all target PEs in the team. This means that symmetric heap objects on each PE are directly load/store accessible by all PEs in the team. See team-management for more detail about its use.
>
> `TEAM_NODE`
>
>
> The team of PEs that are on the same node
>
> `TEAM_SAME_MYPE_NODE`
>
>
> The team of PEs that are the same PE within a node - that is to say, all PEs for which `nvshmem.core.team_my_pe(Teams.TEAM_NODE)` returns the same value.
>
> `TEAM_SAME_GPU`
>
>
> The team of PEs that are on the same GPU
>
> `TEAM_GPU_LEADERS`
>
>
> The team of PEs that are leaders of their respective GPUs

## Initialization Methods

NVSHMEM supports multiple bootstrap methods to initialize the runtime. You must explicitly specify one of the following using the initializer_method argument in `nvshmem.core.init()`.

**Supported methods:**

  * `"mpi"`: Initializes NVSHMEM using an MPI communicator (`mpi4py` is required).
  * `"uid"`: Initializes NVSHMEM using a user-provided unique identifier and rank information.
  * `"emulated_mpi"`: Uses MPI to broadcast a unique ID internally before doing UID-based init.

`nvshmem.core. init ( device: cuda.core._device.Device = None , uid: nvshmem.bindings.nvshmem.uniqueid = None , rank: int = None , nranks: int = None , mpi_comm: None = None , initializer_method: str = '' ) → None`

Initialize the NVSHMEM runtime with either MPI or UID-based bootstrapping.

Args:

  * device (`cuda.core.Device`, required): A Device() that will be bound to this process. All NVSHMEM operations on this process will use this Device
  * uid (`nvshmem.UniqueID`, optional): A unique identifier used for UID-based initialization.
    Must be provided if initializer_method is “uid”.
  * rank (int, optional): Rank of the calling process in the NVSHMEM job. Required for UID-based init.
  * nranks (int, optional): Total number of NVSHMEM ranks in the job. Required for UID-based init.
  * mpi_comm (`mpi4py.MPI.Comm`, optional): MPI communicator to use for MPI-based initialization.
    Defaults to `MPI.COMM_WORLD` if `None` and `initializer_method` is “mpi”.
  * initializer_method (str): Specifies the initialization method. Must be either “mpi” or “uid”.

Raises:

  * NvshmemInvalid: If an invalid initialization method is provided, or required arguments
    for the selected method are missing or incorrect.
  * NvshmemError: If NVSHMEM fails to initialize using the specified method.

Notes:

  * If using MPI-based init, ensure `mpi4py` is compiled against the same MPI distribution you’re running with. A mismatch can result in undefined behavior.
  * For UID-based init, the user is responsible for distributing the uid to all processes and passing in the correct `rank` and `nranks`.
  * UID-based init is useful for bootstrapping over non-MPI runtimes or custom transports.
  * Internally, this sets up a `bindings.InitAttr()` structure which is passed to the NVSHMEM host library.
  * This function uses the `cuda.core` Pathfinder module to locate and load the NVSHMEM host library. The documentation for this function, including the search order can be found at https://nvidia.github.io/cuda-python/cuda-pathfinder/latest/generated/cuda.pathfinder.load_nvidia_dynamic_lib.html#cuda.pathfinder.load_nvidia_dynamic_lib

Example:

    >>> from mpi4py import MPI
    >>> import nvshmem.core as nvshmem
    >>> nvshmem.init(mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
    # OR for UID mode
    >>> uid = nvshmem.get_unique_id()
    >>> nvshmem.init(uid=uid, rank=0, nranks=1, initializer_method="uid")

`nvshmem.core. init_status ( ) → nvshmem.bindings.nvshmem.Init_status`

Get the current initialization status

Returns:
    InitStatus: An enum representing the status of initialization.

## Querying Initialization Status

`class nvshmem.core. InitStatus ( IntEnum )`

NVSHMEM4Py enumerator for initialization status.

> `STATUS_NOT_INITIALIZED`
>
>
> The program is not initialized.
>
> `STATUS_IS_BOOTSTRAPPED`
>
>
> The group of PEs is bootstrapped, but NVSHMEM is not initialized. This means processes can communicate with each other, but CUDA devices are not yet bound to a specific PE. After calling `nvshmem.core.finalize()` (after a successful initialization), the program will be in this state.
>
> `STATUS_IS_INITIALIZED`
>
>
> The NVSHMEM runtime is initialized. After a succesful call to `nvshmem.core.init()`, the program will be in this state.
>
> `STATUS_LIMITED_MPG`
>
>
> The NVSHMEM runtime is initialized with limited MPG support. See https://docs.nvidia.com/nvshmem/api/using.html?highlight=mpg for more details.
>
> `STATUS_FULL_MPG`
>
>
> The NVSHMEM runtime is initialized with full MPG support. See https://docs.nvidia.com/nvshmem/api/using.html?highlight=mpg for more details.
>
> `STATUS_INVALID`
>
>
> The program has an invalid state. This is typically due to an error in the initialization process.

## Finalization

When NVSHMEM operations are complete, call `nvshmem.core.finalize()` to clean up runtime resources.

`nvshmem.core. finalize ( ) → None`

Finalize the NVSHMEM runtime.

This function wraps the NVSHMEM finalization routine. It should be called after all NVSHMEM operations are complete and before the application exits.

Typically, this is called once per process to clean up NVSHMEM resources.

Raises:
    `NvshmemError`: If the NVSHMEM finalization fails.
Example:

    >>> nvshmem.core.finalize()

## Retrieving Version Information

You can query NVSHMEM version details using:

`nvshmem.core. get_version ( ) → nvshmem.core.nvshmem_types.Version`

Get the NVSHMEM4Py version

Returns an object of type nvshmem.core.Version which is a Python class This class contains several strings which represent versions related to NVSHMEM

`Version.openshmem_spec_version` is the OpenSHMEM Spec that this NVSHMEM was built against

`Version.nvshmem4py_version` is the version of the NVSHMEM4Py python library

`Version.libnvshmem_version` is the version of NVSHMEM library that this package has opened

## Retrieving a Unique ID

For UID-based initialization, use:

`nvshmem.core. get_unique_id ( empty=False ) → nvshmem.bindings.nvshmem.uniqueid`

Retrieve or create a unique ID used for UID-based NVSHMEM initialization.

This function wraps the underlying NVSHMEM binding for obtaining a unique ID required when using the `uid` initializer method. Only a single rank (typically rank 0) should call this with empty=False to generate the ID. Other ranks should call it with `empty=True` and receive the ID through a user-defined communication mechanism (e.g., MPI broadcast or socket transfer).

Args:

empty (bool): If True, returns an empty (uninitialized) unique ID structure.
    If False, calls the underlying NVSHMEM function to generate a valid unique ID.
Returns:
    `UniqueID`: A UniqueID object containing the generated or empty NVSHMEM unique ID.
Raises:
    `NvshmemError`: If retrieving the unique ID from NVSHMEM fails.
Example:

    >>> if rank == 0:
    ...     uid = nvshmem.core.get_unique_id()
    ... else:
    ...     uid = nvshmem.core.get_unique_id(empty=True)
    ...
    >>> nvshmem.core.init(uid=uid, rank=rank, nranks=size, initializer_method="uid")
