<a id="device-api-setup"></a>

# Device API – Host-Side Setup

## Host-Side Setup

**Host functions and types.** The following are for use in host code: creating and destroying device communicators,
querying properties, and the requirement and property types. The [`ncclDevComm`](#c.ncclDevComm) structure is then passed to
device code.

### ncclDevComm

### type ncclDevComm

A structure describing a device communicator, as created on the host side
using [`ncclDevCommCreate()`](#c.ncclDevCommCreate). The structure is used primarily on the
device side. In general, fields in this struct are considered internal and
should not be accessed by users. An exception is made for the following fields,
which are guaranteed to be stable across NCCL versions:

### int rank

The rank within the communicator.

### int nRanks

The size of the communicator.

### int lsaRank

### int lsaSize

Rank within the local [LSA](../usage/bufferreg.md#device-api-lsa) team and its size (see [Teams](../usage/deviceapi.md#devapi-teams)).

### uint8_t ginContextCount

The number of supported GIN contexts (see [`ncclGin`](device_gin.md#_CPPv47ncclGin); available
since NCCL 2.28.7).

### ncclDevCommCreate

### [ncclResult_t](types.md#c.ncclResult_t) ncclDevCommCreate([ncclComm_t](types.md#c.ncclComm_t) comm, struct [ncclDevCommRequirements](#c.ncclDevCommRequirements) const \*reqs, struct [ncclDevComm](#c.ncclDevComm) \*outDevComm)

Creates a new device communicator (see [`ncclDevComm`](#c.ncclDevComm)) corresponding to the supplied host-side communicator
*comm*.  The result is returned in the *outDevComm* buffer (which needs to be supplied by the caller).  The caller needs
to also provide a filled-in list of requirements via the *reqs* argument (see [`ncclDevCommRequirements`](#c.ncclDevCommRequirements)); the
function will allocate any necessary resources to meet them. It is recommended to call [`ncclCommQueryProperties()`](#c.ncclCommQueryProperties)
before calling the function; the function will fail if the specified requirements are not supported. Since this is a
collective call, every rank in the communicator needs to participate.  If called within a group, *outDevComm* may not be
filled in until `ncclGroupEnd()` has completed.

Note that this is a *host-side* function.

### ncclDevCommDestroy

### [ncclResult_t](types.md#c.ncclResult_t) ncclDevCommDestroy([ncclComm_t](types.md#c.ncclComm_t) comm, struct [ncclDevComm](#c.ncclDevComm) const \*devComm)

Destroys a device communicator (see [`ncclDevComm`](#c.ncclDevComm)) previously created using [`ncclDevCommCreate()`](#c.ncclDevCommCreate) and
releases any allocated resources.  The caller must ensure that no device kernel that uses this device communicator could
be running at the time this function is invoked.

Note that this is a *host-side* function.

### ncclDevCommRequirements

### type ncclDevCommRequirements

A host-side structure specifying the list of requirements when creating device communicators (see
[`ncclDevComm`](#c.ncclDevComm)). Since NCCL 2.29, this struct must be initialized using `NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER`.

### bool lsaMultimem

Specifies whether multimem support is required for all [LSA](../usage/bufferreg.md#device-api-lsa) ranks.

### int lsaBarrierCount

Specifies the number of memory barriers to allocate (see [`ncclLsaBarrierSession`](device_memory.md#_CPPv4I0E21ncclLsaBarrierSession)). These barriers are
necessary to write fused kernel and may be required by building blocks such as those in [Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels](device_reducecopy.md#device-api-reducecopy).

### int railGinBarrierCount

Specifies the number of railed network barriers to allocate (see [`ncclGinBarrierSession`](device_gin.md#_CPPv4I0E21ncclGinBarrierSession);
available since NCCL 2.28.7).

### int barrierCount

Specifies the number of network/rail hybrid barriers to allocate (see `ncclBarrierSession`; available since NCCL 2.28.7).

### int ginSignalCount

Specifies the number of network signals to allocate (see [`ncclGinSignal_t`](device_gin.md#_CPPv415ncclGinSignal_t); available since NCCL 2.28.7).

### int ginCounterCount

Specifies the number of network counters to allocate (see [`ncclGinCounter_t`](device_gin.md#_CPPv416ncclGinCounter_t); available since NCCL 2.28.7).

### bool ginForceEnable

**Deprecated.** Forces GIN (GPU-Initiated Networking) support to be enabled by automatically setting
`ginConnectionType` to `NCCL_GIN_CONNECTION_FULL`. This field is deprecated in favor of explicitly
setting [`ginConnectionType`](#c.ncclDevCommRequirements.ginConnectionType) to the desired value. When set to `true`, it overrides the
`ginConnectionType` field. New code should use [`ginConnectionType`](#c.ncclDevCommRequirements.ginConnectionType) directly instead of this field.
Available since NCCL 2.28.7, deprecated since NCCL 2.29.7.

### [ncclGinConnectionType_t](#c.ncclGinConnectionType_t) ginConnectionType

Specifies the type of GIN (GPU-Initiated Networking) connection to establish for the device communicator.
This field controls whether GIN is enabled and how it is configured. When set to `NCCL_GIN_CONNECTION_FULL`,
GIN is initialized and all ranks connect to all other ranks in the communicator. When set to `NCCL_GIN_CONNECTION_RAIL`,
GIN is initialized and each rank connects to other ranks in the same rail team. If GIN resources are requested via `ginSignalCount`,
`ginCounterCount`, `barrierCount`, or `railGinBarrierCount` while this field is set to
`NCCL_GIN_CONNECTION_NONE`, device communicator creation will fail with `ncclInvalidArgument`.
Available since NCCL 2.29.7.

See [`ncclGinConnectionType_t`](#c.ncclGinConnectionType_t) for possible values.

### ncclDevResourceRequirements_t \*resourceRequirementsList

Specifies a list of resource requirements.  This is best set to NULL for now.

### ncclTeamRequirements_t \*teamRequirementsList

Specifies a list of requirements for particular teams.  This is best set to NULL for now.

### int ginTrafficClass

Specifies the GIN traffic class. See [Quality of Service](../usage/communicators.md#communicators-qos) for more details. Available since NCCL 2.30.3.

### int worldGinBarrierCount

Specifies the number of world network barriers to allocate. Available since NCCL 2.30.3.

### bool ginStrongSignalsRequired

Specifies whether GIN strong signals are required.
Set to false if kernels using this communicator will not use strong signal operations
(such as [`ncclGin_StrongSignalInc`](device_gin.md#_CPPv423ncclGin_StrongSignalInc) and [`ncclGin_StrongVASignalAdd`](device_gin.md#_CPPv425ncclGin_StrongVASignalAdd)).
Default is true. Available since NCCL 2.30.5.

### bool ginVaSignalsRequired

Specifies whether GIN VA signals are required. Set to false if kernels using this communicator
do not use GIN VA signals (such as [`ncclGin_WeakVASignalInc`](device_gin.md#_CPPv423ncclGin_WeakVASignalInc) and [`ncclGin_StrongVASignalAdd`](device_gin.md#_CPPv425ncclGin_StrongVASignalAdd)).
Default is true. Available since NCCL 2.30.5.

### ncclCommQueryProperties

### [ncclResult_t](types.md#c.ncclResult_t) ncclCommQueryProperties([ncclComm_t](types.md#c.ncclComm_t) comm, [ncclCommProperties_t](#c.ncclCommProperties_t) \*props)

Exposes communicator properties by filling in *props*. Before calling this function, *props* must be initialized
using `NCCL_COMM_PROPERTIES_INITIALIZER`. Introduced in NCCL 2.29.

Note that this is a *host-side* function.

### ncclCommProperties_t

### type ncclCommProperties_t

A structure describing the properties of the communicator. Introduced in NCCL 2.29. Properties include:

### int rank

Rank within the communicator.

### int nRanks

Size of the communicator.

### int cudaDev

CUDA device index.

### int nvmlDev

NVML device index.

### bool deviceApiSupport

Whether the device API is supported. If false, a [`ncclDevComm`](#c.ncclDevComm) cannot be created.

### bool multimemSupport

Whether ranks in the same [LSA](../usage/bufferreg.md#device-api-lsa) team can communicate using multimem. If false,
a [`ncclDevComm`](#c.ncclDevComm) cannot be created with multimem resources.

### [ncclGinType_t](#c.ncclGinType_t) ginType

The GIN type supported by the communicator. If equal to `NCCL_GIN_TYPE_NONE`, a
[`ncclDevComm`](#c.ncclDevComm) cannot be created with GIN connection type `NCCL_GIN_CONNECTION_FULL`.

### int nLsaTeams

The number of [LSA](../usage/bufferreg.md#device-api-lsa) teams across the entire communicator. Available since NCCL 2.29.7.

### [ncclGinType_t](#c.ncclGinType_t) railedGinType

The railed GIN type supported by the communicator. If equal to `NCCL_GIN_TYPE_NONE`, a
[`ncclDevComm`](#c.ncclDevComm) cannot be created with GIN connection type `NCCL_GIN_CONNECTION_RAIL`.
Available since NCCL 2.29.7.

### ncclGinType_t

### type ncclGinType_t

GIN type. Communication between different GIN types is not supported. Possible values include:

### NCCL_GIN_TYPE_NONE

GIN is not supported.

### NCCL_GIN_TYPE_PROXY

Host Proxy GIN type.

### NCCL_GIN_TYPE_GDAKI

GPUDirect Async Kernel-Initiated (GDAKI) GIN type.

### NCCL_GIN_TYPE_GPI

GPU-Push Interface (GPI) GIN type. Requires SpectrumX - see
SpectrumX documentation for details. Added as an experimental
feature in NCCL 2.30.6.

### ncclGinConnectionType_t

### type ncclGinConnectionType_t

Specifies the type of GIN connection for device communicators. This enum controls whether GIN (GPU-Initiated
Networking) resources should be allocated and what connection type to use. Used in [`ncclDevCommRequirements`](#c.ncclDevCommRequirements)
when creating device communicators. Available since NCCL 2.29.7.

### NCCL_GIN_CONNECTION_NONE

No GIN connectivity.

### NCCL_GIN_CONNECTION_FULL

Full GIN connectivity. Each rank is connected to all other ranks.

### NCCL_GIN_CONNECTION_RAIL

Railed GIN connectivity. Each rank is connected to other ranks in the same rail team.

<a id="device-api-host-functions"></a>

## Host-Accessible Device Pointer Functions

**Host functions.** The following are callable from host code only. They provide host-side access to device pointer
functionality, enabling host code to obtain pointers to [LSA](../usage/bufferreg.md#device-api-lsa) memory regions.

All functions return `ncclResult_t` error codes. On success, `ncclSuccess` is returned.
On failure, appropriate error codes are returned (e.g., `ncclInvalidArgument` for invalid parameters,
`ncclInternalError` for internal failures), unless otherwise specified.

The returned pointers are valid for the lifetime of the window.
Pointers should not be used after either the window or communicator is destroyed.
Obtained pointers are device pointers.

### ncclGetLsaMultimemDevicePointer

### [ncclResult_t](types.md#c.ncclResult_t) ncclGetLsaMultimemDevicePointer([ncclWindow_t](types.md#c.ncclWindow_t) window, size_t offset, void \*\*outPtr)

Returns a multimem base pointer for the [LSA](../usage/bufferreg.md#device-api-lsa) team associated with the given window.
This function provides host-side access to the multimem memory functionality.

*window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window.
*outPtr* is the output parameter for the multimem pointer (must not be NULL).

This function requires [LSA](../usage/bufferreg.md#device-api-lsa) multimem support (multicast capability on the system). The window must be registered
with a communicator that supports symmetric memory, and the hardware must support NVLink SHARP multicast functionality.

#### NOTE
If the system does not support multimem, the function returns `ncclSuccess` with `*outPtr` set to `nullptr`.
This allows applications to gracefully detect and handle the absence of multimem support without breaking
the communicator. Users should check if the returned pointer is `nullptr` to determine availability.

Example:

```C
void* multimemPtr;
ncclResult_t result = ncclGetLsaMultimemDevicePointer(window, 0, &multimemPtr);
if (result == ncclSuccess) {
    if (multimemPtr != nullptr) {
        // Use multimemPtr for multimem operations
    } else {
        // Multimem not supported, use fallback approach
    }
}
```

### ncclGetMultimemDevicePointer

### [ncclResult_t](types.md#c.ncclResult_t) ncclGetMultimemDevicePointer([ncclWindow_t](types.md#c.ncclWindow_t) window, size_t offset, ncclMultimemHandle multimem, void \*\*outPtr)

Returns a multimem base pointer using a provided multimem handle instead of the window’s internal multimem.
This function enables using external or custom multimem handles for pointer calculation.

*window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window.
*multimem* is the multimem handle containing the multimem base pointer (multimem.mcBasePtr must not be NULL).
*outPtr* is the output parameter for the multimem pointer (must not be NULL).

This function requires [LSA](../usage/bufferreg.md#device-api-lsa) multimem support (multicast capability on the system).

#### NOTE
If the system does not support multimem, the function returns `ncclSuccess` with `*outPtr` set to `nullptr`.
The function validates that `multimem.mcBasePtr` is not nullptr before proceeding.

Example:

```C
// Get multimem handle from device communicator setup
ncclMultimemHandle customHandle;
// ... (obtain handle)

void* multimemPtr;
ncclResult_t result = ncclGetMultimemDevicePointer(window, 0, customHandle, &multimemPtr);
if (result == ncclSuccess) {
    if (multimemPtr != nullptr) {
        // Use multimemPtr for multimem operations with custom handle
    } else {
        // Multimem not supported, use fallback approach
    }
}
```

### ncclGetLsaDevicePointer

### [ncclResult_t](types.md#c.ncclResult_t) ncclGetLsaDevicePointer([ncclWindow_t](types.md#c.ncclWindow_t) window, size_t offset, int lsaRank, void \*\*outPtr)

Returns a load/store accessible pointer to the memory buffer of a specific [LSA](../usage/bufferreg.md#device-api-lsa) peer
within the window. This function provides host-side access to LSA pointer functionality using LSA rank directly.

*window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window
(must be >= 0 and < window size).
*lsaRank* is the LSA rank of the target peer (must be >= 0 and < LSA team size).
*outPtr* is the output parameter for the LSA pointer (must not be NULL).

On success, `ncclSuccess` is returned and the LSA pointer is returned in `outPtr`.

The window must be registered with a communicator that supports LSA. The LSA rank must be within the valid range
for the LSA team, and the target peer must be load/store accessible (P2P connectivity required).

Example:

```C
void* lsaPtr;
ncclResult_t result = ncclGetLsaDevicePointer(window, 0, 1, &lsaPtr);
if (result == ncclSuccess) {
    // Use lsaPtr to access LSA peer 1's memory
}
```

### ncclGetPeerDevicePointer

### [ncclResult_t](types.md#c.ncclResult_t) ncclGetPeerDevicePointer([ncclWindow_t](types.md#c.ncclWindow_t) window, size_t offset, int peer, void \*\*outPtr)

Returns a load/store accessible pointer to the memory buffer of a specific world rank peer within the window.
This function converts world rank to [LSA](../usage/bufferreg.md#device-api-lsa) rank internally and provides host-side access
to peer pointer functionality.

*window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window.
*peer* is the world rank of the target peer (must be >= 0 and < communicator size).
*outPtr* is the output parameter for the peer pointer (must not be NULL).

On success, `ncclSuccess` is returned and the peer pointer is returned in `outPtr`.

If the peer is not reachable via [LSA](../usage/bufferreg.md#device-api-lsa) (not in LSA team), `outPtr` is set to NULL and
`ncclSuccess` is returned.
This matches the behavior of the device-side `ncclGetPeerPointer` function.

The window must be registered with a communicator that supports LSA. The peer rank must be within the valid range
for the communicator, and the target peer must be load/store accessible (P2P connectivity required).

Example:

```C
void* peerPtr;
ncclResult_t result = ncclGetPeerDevicePointer(window, 0, 2, &peerPtr);
if (result == ncclSuccess) {
    if (peerPtr != NULL) {
        // Use peerPtr to access world rank 2's memory
    } else {
        // Peer 2 is not reachable via LSA
    }
}
```
