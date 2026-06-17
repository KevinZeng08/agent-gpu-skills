# Device Communicator Setup

Host-side methods and resources for creating an NCCL device communicator.
The device-side communication primitives themselves are available only
from CUDA kernels and are documented under the C device API
([Device API](../../api/device.md)); this page covers what the Python (host) side
exposes for bootstrapping them. The configuration object passed to
[`Communicator.create_dev_comm()`](#nccl.core.Communicator.create_dev_comm) is documented in
[Configuration](../configuration.md).

## create_dev_comm

#### Communicator.create_dev_comm(requirements: [NCCLDevCommRequirements](../configuration.md#nccl.core.NCCLDevCommRequirements) | None = None) → [DevCommResource](../resources.md#nccl.core.DevCommResource)

Creates a device communicator for device-side NCCL operations.

Device communicators enable direct GPU kernel access to NCCL
communication primitives. Multiple device communicators can be
created from one host communicator. The returned
[`DevCommResource`](../resources.md#nccl.core.DevCommResource) is tracked by the
communicator and may be released explicitly via its
`close()` method, or automatically
when the communicator is destroyed or aborted. Access the device
communicator pointer via [`DevCommResource.ptr`](../resources.md#nccl.core.DevCommResource.ptr) or
`resource.dev_comm.ptr`.

* **Parameters:**
  **requirements** – Configuration for device communicator resource
  allocation. If `None`, a default
  [`NCCLDevCommRequirements`](../configuration.md#nccl.core.NCCLDevCommRequirements) is used.
  Defaults to `None`.
* **Returns:**
  [`DevCommResource`](../resources.md#nccl.core.DevCommResource) for the device
  communicator.
* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

#### SEE ALSO
[`ncclDevCommCreate()`](../../api/device_setup.md#c.ncclDevCommCreate)

## GIN type enums

GPU Interconnect Network (GIN) enums describing what device-side network
transport is available on a communicator and which connection topology
the user requires.

### NcclGinType

### *class* nccl.core.NcclGinType(value, names=<not given>, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: `IntEnum`

GIN transport type, mirroring [`ncclGinType_t`](../../api/device_setup.md#c.ncclGinType_t).

Reported by [`Communicator.gin_type`](class.md#nccl.core.Communicator.gin_type) and
[`Communicator.railed_gin_type`](class.md#nccl.core.Communicator.railed_gin_type) to indicate which device-side
network transport, if any, is available on the communicator.

#### NONE *= 0*

GIN not available on this communicator.

#### PROXY *= 2*

Proxy-based GIN. Network operations issued from a device kernel are
relayed through a CPU proxy thread.

#### GDAKI *= 3*

GPUDirect Async Kernel-Initiated (GDA-KI). The kernel directly
issues network operations to the NIC, bypassing the CPU proxy.

#### GPI *= 4*

GPU-Push Interface. GPU threads push network descriptors directly
to a NIC-visible MMIO queue, with no CPU involvement and no memory
barriers.

### NcclGinConnectionType

### *class* nccl.core.NcclGinConnectionType(value, names=<not given>, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: `IntEnum`

GIN connection topology, mirroring [`ncclGinConnectionType_t`](../../api/device_setup.md#c.ncclGinConnectionType_t).

Set on the `gin_connection_type` field of
[`NCCLDevCommRequirements`](../configuration.md#nccl.core.NCCLDevCommRequirements) before calling
[`Communicator.create_dev_comm()`](#nccl.core.Communicator.create_dev_comm) to declare which peers must be
reachable via GIN from device code.

#### NONE *= 0*

No GIN connection requested.

#### FULL *= 1*

Fully connected. Every rank in the communicator must be reachable
from every other rank via GIN.

#### RAIL *= 2*

Rail-restricted. Ranks must be reachable via GIN only within the
same rail (network plane).
