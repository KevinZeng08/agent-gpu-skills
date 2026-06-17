# Communicator Class

### *class* nccl.core.Communicator(ptr: int = 0)

Bases: `object`

NCCL communicator for collective and point-to-point operations.

A communicator represents a group of ranks that can perform collective
operations (e.g. allreduce, broadcast) and point-to-point operations
(send/recv). Each rank has a unique ID in `[0, nranks)`.

Communicator instances expose a number of properties for inspection
(`ptr`, `nranks`, `device`, `rank`, plus device-API related
properties like `cuda_dev`, `nvml_dev`, `device_api_support`,
`multimem_support`, `gin_type`, `n_lsa_teams`,
`host_rma_support`, `railed_gin_type`); see the per-property
documentation for details.

#### \_\_init_\_(ptr: int = 0) → None

Initializes a communicator with a raw NCCL pointer.

Unlike the [`init()`](lifecycle.md#nccl.core.Communicator.init) classmethod, this constructor allows
`ptr=0` for creating null communicators (e.g. when [`split()`](lifecycle.md#nccl.core.Communicator.split)
excludes a rank). A null communicator can later be initialized via
[`initialize()`](lifecycle.md#nccl.core.Communicator.initialize), or used as the caller for [`grow()`](lifecycle.md#nccl.core.Communicator.grow) to
join an existing communicator.

* **Parameters:**
  **ptr** – Integer representing an NCCL communicator pointer (0 for a
  null communicator). Defaults to 0.

## Properties

### Identity

#### Communicator.ptr

Raw pointer to the underlying [`ncclComm_t`](../../api/types.md#c.ncclComm_t) structure
(0 if destroyed or null).

#### Communicator.is_valid

Whether the communicator is valid (not destroyed or null).

#### Communicator.nranks

Total number of ranks in the communicator.

* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

#### Communicator.device

CUDA device associated with this communicator.

Returns a [`cuda.core.Device`](https://nvidia.github.io/cuda-python/cuda-core/latest/generated/cuda.core.Device.html#cuda.core.Device) providing additional
functionality such as `to_system_device` for obtaining the NVML
device, device properties, and synchronization.

* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

#### Communicator.rank

This caller’s rank within the communicator (0 to nranks - 1).

* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

### Device-API capability

These properties reflect the underlying NCCL [`ncclCommProperties_t`](../../api/device_setup.md#c.ncclCommProperties_t)
structure.

#### Communicator.cuda_dev

CUDA device ID associated with this communicator.

* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

#### Communicator.nvml_dev

NVML device ID for the GPU associated with this communicator.

Uses the NVML indexing space, which may differ from CUDA indexing.

* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

#### Communicator.device_api_support

Whether device-side NCCL operations are supported on this platform.

If False, a device communicator cannot be created.

* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

#### Communicator.multimem_support

Whether ranks in the same LSA team can communicate using multimem.

If False, a device communicator cannot be created with multimem
resources.

* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

#### Communicator.gin_type

GPU Interconnect Network (GIN) type.

If equal to NcclGinType.NONE, a device communicator cannot be
created with GIN resources.

* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

#### Communicator.n_lsa_teams

Number of Local Shared Array (LSA) teams for this communicator.

* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

#### Communicator.host_rma_support

Whether host RMA is supported on this communicator.

* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.

#### Communicator.railed_gin_type

Railed GIN type supported by this communicator.

If equal to [`NcclGinType.NONE`](device_setup.md#nccl.core.NcclGinType.NONE), a device communicator
cannot be created with GIN connection type
[`NcclGinConnectionType.RAIL`](device_setup.md#nccl.core.NcclGinConnectionType.RAIL).

* **Raises:**
  [**NcclInvalid**](../types.md#nccl.core.NcclInvalid) – If the communicator is not initialized.
