# Environment Variables

## Standard options

`NVSHMEM_VERSION`

_Type: bool_

_Default: false_

Print library version at startup.

`NVSHMEM_INFO`

_Type: bool_

_Default: false_

Print environment variable options at startup.

`NVSHMEM_SYMMETRIC_SIZE`

_Type: size_

_Default: 1073741824_

Specifies the size (in bytes) of the symmetric heap memory per PE. The resulting size is implementation-defined and must be at least as large as the integer ceiling of the product of the numeric prefix and the scaling factor. The allowed character suffixes for the scaling factor are as follows:

>   * k or K multiplies by 2^10 (kibibytes)
>   * m or M multiplies by 2^20 (mebibytes)
>   * g or G multiplies by 2^30 (gibibytes)
>   * t or T multiplies by 2^40 (tebibytes)
>

For example, string ‘20m’ is equivalent to the integer value 20971520, or 20 mebibytes. Similarly the string ‘3.1M’ is equivalent to the integer value 3250586\. Only one multiplier is recognized and any characters following the multiplier are ignored, so ‘20kk’ will not produce the same result as ‘20m’. Usage of string ‘.5m’ will yield the same result as the string ‘0.5m’. An invalid value for `NVSHMEM_SYMMETRIC_SIZE` is an error, which the NVSHMEM library shall report by either returning a nonzero value from `nvshmem_init_thread` or causing program termination.

`NVSHMEM_DEBUG`

_Type: string_

_Default: “”_

Set to enable debugging messages. Optional values: VERSION, WARN, INFO, ABORT, TRACE

## Bootstrap options

`NVSHMEM_BOOTSTRAP`

_Type: string_

_Default: “PMI”_

Name of the default bootstrap that should be used to initialize NVSHMEM. Allowed values: PMI, MPI, SHMEM, plugin

`NVSHMEM_BOOTSTRAP_PMI`

_Type: string_

_Default: “PMI”_

Name of the PMI bootstrap that should be used to initialize NVSHMEM. Allowed values: PMI, PMI-2, PMIX

`NVSHMEM_BOOTSTRAP_PLUGIN`

_Type: string_

_Default: “”_

Name of the bootstrap plugin file to load when NVSHMEM_BOOTSTRAP=plugin is specified.

`NVSHMEM_BOOTSTRAP_MPI_PLUGIN`

_Type: string_

_Default: “nvshmem_bootstrap_mpi.so”_

Name of the MPI bootstrap plugin file.

`NVSHMEM_BOOTSTRAP_SHMEM_PLUGIN`

_Type: string_

_Default: “nvshmem_bootstrap_shmem.so”_

Name of the SHMEM bootstrap plugin file.

`NVSHMEM_BOOTSTRAP_PMI_PLUGIN`

_Type: string_

_Default: “nvshmem_bootstrap_pmi.so”_

Name of the PMI bootstrap plugin file.

`NVSHMEM_BOOTSTRAP_PMI2_PLUGIN`

_Type: string_

_Default: “nvshmem_bootstrap_pmi2.so”_

Name of the PMI-2 bootstrap plugin file.

`NVSHMEM_BOOTSTRAP_PMIX_PLUGIN`

_Type: string_

_Default: “nvshmem_bootstrap_pmix.so”_

Name of the PMIx bootstrap plugin file.

`NVSHMEM_BOOTSTRAP_UID_PLUGIN`

_Type: string_

_Default: “nvshmem_bootstrap_uid.so”_

Name of the UID bootstrap plugin file.

`NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME`

_Type: string_

_Default: “”_

Define to a list of prefixes to filter interfaces to be used by NVSHMEM. Using the `^` symbol, NVSHMEM will exclude interfaces starting with any prefix in that list. To match (or not) an exact interface name instead of a prefix, prefix the string with the `=` character.

Examples: `eth` : Use all interfaces starting with `eth`, e.g. `eth0`, `eth1`, … `=eth0` : Use only interface `eth0` `^docker` : Do not use any interface starting with `docker` `^=docker0` : Do not use interface `docker0`.

Note: By default, the loopback interface (`lo`) and docker interfaces (`docker*`) would not be selected unless there are no other interfaces available. If you prefer to use `lo` or `docker*` over other interfaces, you would need to explicitly select them using `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME`. The default algorithm will also favor interfaces starting with `ib` over others. Setting `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME` will bypass the automatic interface selection algorithm and may use all interfaces matching the manual selection.

`NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY`

_Type: string_

_Default: “AF_INET”_

Name of the socket family that interface belongs to. Allowed values: AF_INET6, AF_INET.

`NVSHMEM_BOOTSTRAP_UID_SESSION_ID`

_Type: string_

_Default: “”_

Name of the UID session identifier, as specified by a combination of <ipv4>:<TCP port> or [<ipv6>]:<TCP port> or <hostname>:<TCP port>.

`NVSHMEM_BOOTSTRAP_SHMEM_MODE`

_Type: string_

_Default: “auto”_

Select the OpenSHMEM bootstrap collective path. Allowed values: `auto`, `legacy`, `teams`.

## Additional options

`NVSHMEM_DEBUG_FILE`

_Type: string_

_Default: “”_

Debugging output filename, may contain %h for hostname and %p for pid.

`NVSHMEM_DEBUG_ATTACH_DELAY`

_Type: int_

_Default: 0_

Delay, in seconds, during the first NVSHMEM initialization call to allow for attaching a debugger.

`NVSHMEM_MAX_TEAMS`

_Type: long_

_Default: 128_

Maximum number of simultaneous teams allowed. This limit includes both user-visible teams and internal teams created by NVSHMEM. For multi-CTA collectives, NVSHMEM creates internal teams corresponding to reserved and user-created teams. With NVLS enabled, each newly created team can require up to 48 internal teams. Set `NVSHMEM_MAX_TEAMS` large enough to accommodate these internal teams; NVSHMEM reports an error at runtime if the limit is insufficient.

`NVSHMEM_MAX_MEMORY_PER_GPU`

_Type: size_

_Default: 137438953472_

Maximum memory per GPU

`NVSHMEM_DISABLE_CUDA_VMM`

_Type: bool_

_Default: false_

Disable use of CUDA VMM for P2P memory mapping. By default, CUDA VMM is enabled on x86 and disabled on P9. CUDA VMM feature in NVSHMEM requires CUDA RT version and CUDA Driver version to be greater than or equal to 11.3.

`NVSHMEM_DISABLE_P2P`

_Type: bool_

_Default: false_

Disable P2P connectivity of GPUs even when available.

`NVSHMEM_DISABLE_MNNVL`

_Type: bool_

_Default: false_

Disable MNNVL connectivity of GPUs even when available.

`NVSHMEM_DISABLE_NVLS`

_Type: bool_

_Default: false_

Disable NVLINK SHARP collectives for P2P connected GPUs over NVSwitch even when available.

`NVSHMEM_ENABLE_LOGICAL_ENDPOINT`

_Type: bool_

_Default: false_

Enable logical endpoint support.

`NVSHMEM_CUMEM_GRANULARITY`

_Type: size_

_Default: 536870912_

Granularity for `cuMemAlloc`/`cuMemCreate`.

`NVSHMEM_CUDA_LIMIT_STACK_SIZE`

_Type: size_

_Default: 0_

Specify limit on stack size of each GPU thread on P9.

`NVSHMEM_CUDA_PATH`

_Type: string_

_Default: “”_

Path to directory containing `libcuda.so` for use when not in the default location.

`NVSHMEM_PROXY_REQUEST_BATCH_MAX`

_Type: int_

_Default: 32_

Maxmum number of requests that the proxy thread processes in a single iteration of the progress loop.

`NVSHMEM_G_BUF_SIZE`

_Type: int_

_Default: 4194304_

Size of the `g_buf` used to perform `shmem_g` operations in parallel. Must be a multiple of 16 bytes.

`NVSHMEM_G_COALESCING_BUF_SIZE`

_Type: int_

_Default: 67108864_

Size of the buffer used for coalescing `shmem_g` operations. Must be a multiple of 256 bytes. NVSHMEM requires its value to be `NVSHMEM_G_BUF_SIZE * 16`.

`NVSHMEM_MAX_PEER_STREAMS`

_Type: int_

_Default: 16_

Maximum number of CUDA streams per node.

`NVSHMEM_CPU_AFFINITY`

_Type: string_

_Default: “AUTO”_

Controls NUMA-aware CPU affinity pinning during initialization. Allowed values: `AUTO` enables automatic NUMA-local pinning; `OFF` disables pinning. `AUTO` only narrows current affinity and does not overwrite existing settings.

## Collectives options

`NVSHMEM_DISABLE_NCCL`

_Type: bool_

_Default: false_

Disable use of NCCL for collective operations.

`NVSHMEM_BARRIER_DISSEM_KVAL`

_Type: int_

_Default: 2_

Radix of the dissemination algorithm used for barriers.

`NVSHMEM_BARRIER_TG_DISSEM_KVAL`

_Type: int_

_Default: 2_

Radix of the dissemination algorithm used for thread group barriers.

`NVSHMEM_FCOLLECT_LL_THRESHOLD`

_Type: size_

_Default: 2048_

Message size threshold up to which fcollect LL algo will be used.

`NVSHMEM_BCAST_ALGO`

_Type: int_

_Default: 0_

Broadcast algorithm to be used.

  * 0 - use default algorithm selection strategy

`NVSHMEM_REDMAXLOC_ALGO`

_Type: int_

_Default: 1_

Reduction algorithm to be used.

  * 1 - default, flag alltoall algorithm
  * 2 - flat reduce + flat bcast
  * 3 - topo-aware two-level reduce + topo-aware bcast

`NVSHMEM_REDUCE_SCRATCH_SIZE`

_Type: size_t_

_Default: 524288_

Amount of symmetric heap memory (minimum 16B, multiple of 8B) reserved by runtime for every team to implement reduce and reducescatter collectives.

## Transport options

`NVSHMEM_REMOTE_TRANSPORT`

_Type: string_

_Default: “ibrc”_

Selected transport for remote operations: ibrc, ucx, libfabric, ibdevx, gpunetio, none.

`NVSHMEM_DISABLE_IB_NATIVE_ATOMICS`

_Type: bool_

_Default: false_

Disable use of InfiniBand native atomics.

`NVSHMEM_DISABLE_GDRCOPY`

_Type: bool_

_Default: false_

Disable use of GDRCopy in IB RC Transport.

`NVSHMEM_ENABLE_NIC_PE_MAPPING`

_Type: bool_

_Default: false_

When not set or set to 0, a PE is assigned to the NIC on the node that is closest to it by distance. When set to 1, NVSHMEM either assigns NICs to PEs on a round-robin basis or uses `NVSHMEM_HCA_PE_MAPPING` or `NVSHMEM_HCA_LIST` when they are specified.

`NVSHMEM_NETDEVS_POLICY`

_Type: string_

_Default: “AUTO”_

Policy for automatic NIC assignment when `NVSHMEM_ENABLE_NIC_PE_MAPPING` is 0. `AUTO` preserves the default NVSHMEM behavior and balances over local NVSHMEM PEs. `EXTERNAL_SHARING_PCIE_SWITCH_NIC_EXCLUSIVE` balances over all node-local GPUs to avoid external NVSHMEM instances sharing PCIe-switch-local NICs when topology permits; NICs may still be shared when there are fewer NICs than GPUs.

`NVSHMEM_TRANSPORT_BATCH_MAX_OPS`

_Type: int_

_Default: 16_

Maximum number of consecutive proxy requests to mark with a transport batching hint before ending the current transport-level batch. Applies only to transports and operation types that implement batching hints.

`NVSHMEM_IB_GID_INDEX`

_Type: int_

_Default: -1_

Source GID Index for ROCE. By default, it would dynamically discover the GID supported by the NIC.

`NVSHMEM_IB_TRAFFIC_CLASS`

_Type: int_

_Default: 0_

Traffic calss for ROCE.

`NVSHMEM_IB_SL`

_Type: int_

_Default: 0_

Service level to use over IB/ROCE.

`NVSHMEM_IB_ADDR_FAMILY`

_Type: string_

_Default: AF_INET_

IP address family associated to GID dynamically selected by NVSHMEM when `NVSHMEM_IB_GID_INDEX` is left unset.

`NVSHMEM_IB_ADDR_RANGE`

_Type: string_

_Default: ::/0_

Defines the range of valid GIDs dynamically selected by NVSHMEM when `NVSHMEM_IB_GID_INDEX` is left unset.

`NVSHMEM_IB_ROCE_VERSION_NUM`

_Type: int_

_Default: 2_

ROCE version associated to IB GID dynamically selected by NVSHMEM when `NVSHMEM_IB_GID_INDEX` is left unset.

`NVSHMEM_IB_TIMEOUT`

_Type: int_

_Default: 20_

QP acknowledgement timeout for IB transports. Valid range: 0-31.

`NVSHMEM_IB_RETRY_CNT`

_Type: int_

_Default: 7_

QP retry count for IB transports. Valid range: 0-7.

`NVSHMEM_IB_PKEY_INDEX`

_Type: int_

_Default: 0_

Partition key (pkey) index to use for InfiniBand transport queue pairs. The default is 0, the default partition.

`NVSHMEM_IB_ENABLE_RELAXED_ORDERING`

_Type: bool_

_Default: true_

Enable PCIe relaxed ordering on transports over IB/ROCE, such as IBRC, IBGDA, and IBDEVX.

`NVSHMEM_IB_NUM_RC_PER_DEVICE`

_Type: int_

_Default: 1_

Number of RC QPs to create per device in the IB proxy-based transports. A device is each enumerated IB device, either a full HCA or a single port of a multi-port HCA.

`NVSHMEM_HCA_PREFIX`

_Type: string_

_Default: “mlx5”_

Prefix of HCA interface names. Example, mlx5, ibp.

`NVSHMEM_HCA_LIST`

_Type: string_

_Default: “”_

Comma-separated list of HCAs to use in the NVSHMEM application. Entries are of the form `hca_name:port`, e.g. `mlx5_1:1,mlx5_2:2` and entries prefixed by ^ are excluded. `NVSHMEM_ENABLE_NIC_PE_MAPPING` must be set to 1 for this variable to be effective.

`NVSHMEM_HCA_PE_MAPPING`

_Type: string_

_Default: “”_

Specifies mapping of HCAs to PEs as a comma-separated list. Each entry in the comma separated list is of the form `hca_name:port:count`. For example, `mlx5_0:1:2,mlx5_0:2:2` indicates that PE0, PE1 are mapped to port 1 of mlx5_0, and PE2, PE3 are mapped to port 2 of mlx5_0. `NVSHMEM_ENABLE_NIC_PE_MAPPING` must be set to 1 for this variable to be effective.

`NVSHMEM_DISABLE_LOCAL_ONLY_PROXY`

_Type: bool_

_Default: false_

When running on an NVLink-only configuaration (No-IB, No-UCX), completely disable the proxy thread. This will disable device side global exit and device side wait timeout polling (enabled by `NVSHMEM_TIMEOUT_DEVICE_POLLING` build-time variable) because these are processed by the proxy thread.

`NVSHMEM_TMA_POLICY`

_Type: string_

_Default: “DISABLE”_

Controls TMA usage for device-side point-to-point operations over NVLink. Valid values are `DISABLE`, `ENABLE`, and `FORCE`. `DISABLE` prevents NVSHMEM from using TMA. `ENABLE` allows NVSHMEM to use TMA when the GPU architecture, registered CTA shared memory, topology, and transfer shape support it. `FORCE` requires TMA support to be available during initialization and fails on unsupported devices.

`NVSHMEM_LIBFABRIC_PROVIDER`

_Type: string_

_Default: “cxi”_

Set the feature set provider for the libfabric transport: cxi, efa, verbs

`NVSHMEM_LIBFABRIC_MAX_NIC_PER_PE`

_Type: int_

_Default: 16_

Set the maximum number of NICs per PE for use in the libfabric provider.

`NVSHMEM_LIBFABRIC_PROXY_REQUEST_BATCH_MAX`

_Type: int_

_Default: 32_

Maximum number of requests that the libfabric transport processes per queue in a single iteration of the progress loop.

`NVSHMEM_LIBFABRIC_DISABLE_BATCH_RMA`

_Type: bool_

_Default: false_

Disable support for batched RMA with `FI_MORE`.

`NVSHMEM_LIBFABRIC_SIGNAL_WAIT_SPIN_COUNT`

_Type: int_

_Default: 1024_

Number of polling iterations for pending signal-delivery work in the libfabric transport before sleeping. Increasing this can reduce latency for signal-heavy operations at the cost of CPU usage; 0 sleeps immediately. The value must be non-negative.

`NVSHMEM_IBGDA_NUM_DCT`

_Type: int_

_Default: 2_

Number of DCT QPs used in GPU-initiated communication transport.

`NVSHMEM_IBGDA_NUM_DCI`

_Type: int_

_Default: 1_

Total number of DCI QPs used in GPU-initiated communication transport. Set to 0 or a negative number to use automatic configuration.

`NVSHMEM_IBGDA_NUM_SHARED_DCI`

_Type: int_

_Default: 1_

Number of DCI QPs in the shared pool. The rest of DCI QPs (NVSHMEM_IBGDA_NUM_DCI \- NVSHMEM_IBGDA_NUM_SHARED_DCI) are exclusively assigned. Valid value: [1, NVSHMEM_IBGDA_NUM_DCI].

`NVSHMEM_IBGDA_DCI_MAP_BY`

_Type: string_

_Default: “cta”_

Specifies how exclusive DCI QPs are assigned. Choices are: cta, sm, warp, dct.

  * cta: round-robin by CTA ID (default).
  * sm: round-robin by SM ID.
  * warp: round-robin by Warp ID.
  * dct: round-robin by DCT ID.

`NVSHMEM_IBGDA_NUM_RC_PER_PE`

_Type: int_

_Default: 2_

Number of RC QPs per peer PE used in GPU-initiated communication transport. Set to 0 to disable RC QPs (default 2). If set to a positive number, DCI will be used for enforcing consistency only.

`NVSHMEM_IBGDA_RC_MAP_BY`

_Type: string_

_Default: “cta”_

Specifies how RC QPs are assigned. Choices are: cta, sm, warp.

  * cta: round-robin by CTA ID (default).
  * sm: round-robin by SM ID.
  * warp: round-robin by Warp ID.

`NVSHMEM_IBGDA_FORCE_NIC_BUF_MEMTYPE`

_Type: string_

_Default: “gpumem”_

Force NIC buffer memory type. Valid choices are: gpumem (default), hostmem. For other values, use auto discovery.

`NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH`

_Type: int_

_Default: 32_

Number of requests to be batched before submitting to the NIC. It will be rounded up to the nearest power of 2. Set to 1 for aggressive submission.

`NVSHMEM_IBGDA_NUM_FETCH_SLOTS_PER_DCI`

_Type: int_

_Default: 1024_

Number of internal buffer slots for fetch operations for each DCI QP. It will be rounded up to the nearest power of 2.

`NVSHMEM_IBGDA_NUM_FETCH_SLOTS_PER_RC`

_Type: int_

_Default: 1024_

Number of internal buffer slots for fetch operations for each RC QP. It will be rounded up to the nearest power of 2.

`NVSHMEM_IB_ENABLE_IBGDA`

_Type: bool_

_Default: false_

Set to enable GPU-initiated communication transport.

`NVSHMEM_IBGDA_NIC_HANDLER`

_Type: string_

_Default: auto_

Selects the processor used for ringing NIC’s doorbell. Choices are `auto`, `gpu`, `cpu`, `cpu_cuda_memory`.

`auto`: Use GPU SMs and fallback to CPU if it is not supported (default). `gpu`: Use GPU SMs. `cpu`: Use CPU proxy thread. `cpu_cuda_memory`: Use CPU with CUDA memory.

`NVSHMEM_IB_DISABLE_DMABUF`

_Type: bool_

_Default: false_

Set to disable DMAbuf in any IB based remote transport.

`NVSHMEM_DISABLE_DATA_DIRECT`

_Type: bool_

_Default: false_

Disable use of DirectNIC in IB transport

`NVSHMEM_IBGDA_ENABLE_MULTI_PORT`

_Type: bool_

_Default: false_

Set to enable multiple NICs per PE if available.

`NVSHMEM_GPUNETIO_ENABLE_GDAKI`

_Type: bool_

_Default: false_

Set to enable GPU-initiated communication transport via GPUNetIO. When set to 1, `NVSHMEM_REMOTE_TRANSPORT` must be set to `gpunetio`.

`NVSHMEM_GPUNETIO_NIC_HANDLER`

_Type: string_

_Default: “auto”_

Specifies the processor used for ringing the NIC’s doorbell. Choices are `auto`, `gpu`, `gpu_sm_bf`, `cpu`.

  * `auto`: Use GPU SMs and fallback to CPU if it is not supported.
  * `gpu`: Use GPU SMs, regular doorbell.
  * `gpu_sm_bf`: Use GPU SMs, BlueFlame doorbell.
  * `cpu`: Use CPU.

`NVSHMEM_GPUNETIO_NUM_RC_PER_PE_GPU`

_Type: int_

_Default: 2_

Number of GPU-data-path RC QPs per peer PE in the GPUNetIO transport. This only takes effect when `NVSHMEM_GPUNETIO_ENABLE_GDAKI` is set to 1. Otherwise, the GPU data path is disabled and this value is ignored.

`NVSHMEM_GPUNETIO_NUM_RC_PER_PE_CPU`

_Type: int_

_Default: 2_

Number of CPU-data-path RC QPs per peer PE in the GPUNetIO transport. The CPU data path is always active and does not require GDAKI. This value must be greater than 0.

`NVSHMEM_GPUNETIO_NUM_REQUESTS_IN_BATCH`

_Type: int_

_Default: 32_

Number of requests to be batched before submitting to the NIC when using GDAKI. It will be rounded up to the nearest power of 2. Set to 1 for aggressive submission. Only takes effect when `NVSHMEM_GPUNETIO_ENABLE_GDAKI=1`. This value must be positive and must not be larger than the QP depth.

`NVSHMEM_GPUNETIO_NUM_FETCH_SLOTS_PER_RC`

_Type: int_

_Default: 1024_

Number of internal buffer slots for fetch operations for each RC QP when using GDAKI. It will be rounded up to the nearest power of 2. Only takes effect when `NVSHMEM_GPUNETIO_ENABLE_GDAKI=1`.

`NVSHMEM_GPUNETIO_ENABLE_ORDERING_SEMANTIC`

_Type: bool_

_Default: false_

Set to enable ordering semantic for DDP (Direct Data Placement) mode for GPUNetIO QPs. This depends on the DOCA SDK provided via `DOCA_SDK_LIB_PATH` to GPUNetIO.

## NVTX options

`NVSHMEM_NVTX`

_Type: string_

_Default: “off”_

Set to enable NVTX instrumentation. Accepts a comma separated list of instrumentation groups. By default the NVTX instrumentation is disabled.

    init                : library setup
    alloc               : memory management
    launch              : kernel launch routines
    coll                : collective communications
    wait                : blocking point-to-point synchronization
    wait_on_stream      : point-to-point synchronization (on stream)
    test                : non-blocking point-to-point synchronization
    memorder            : memory ordering (quiet, fence)
    quiet_on_stream     : nvshmemx_quiet_on_stream
    atomic_fetch        : fetching atomic memory operations
    atomic_set          : non-fetchong atomic memory operations
    rma_blocking        : blocking remote memory access operations
    rma_nonblocking     : non-blocking remote memory access operations
    proxy               : activity of the proxy thread
    common              : init,alloc,launch,coll,memorder,wait,atomic_fetch,rma_blocking,proxy
    all                 : all groups
    off                 : disable all NVTX instrumentation
