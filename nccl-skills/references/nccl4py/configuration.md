# Configuration

Configuration objects passed to communicator creation methods, plus the
flag enums they consume.

## NCCLConfig

Used by [`Communicator.init()`](communicator/lifecycle.md#nccl.core.Communicator.init), [`Communicator.split()`](communicator/lifecycle.md#nccl.core.Communicator.split),
[`Communicator.shrink()`](communicator/lifecycle.md#nccl.core.Communicator.shrink), and [`Communicator.grow()`](communicator/lifecycle.md#nccl.core.Communicator.grow). Fields
left unset (`None`) remain at NCCL’s internal default; values are
validated by the C library when the config is consumed.

### *class* nccl.core.NCCLConfig(\*, blocking: bool | None = None, cga_cluster_size: int | None = None, min_ctas: int | None = None, max_ctas: int | None = None, net_name: str | None = None, split_share: bool | None = None, traffic_class: int | None = None, comm_name: str | None = None, collnet_enable: bool | None = None, cta_policy: [CTAPolicy](#nccl.core.CTAPolicy) | None = None, shrink_share: bool | None = None, nvls_ctas: int | None = None, n_channels_per_net_peer: int | None = None, nvlink_centric_sched: bool | None = None, graph_usage_mode: int | None = None, num_rma_ctx: int | None = None, max_p2p_peers: int | None = None, graph_stream_ordering: int | None = None)

Bases: `object`

NCCL configuration for communicator initialization.

Provides configuration options for NCCL communicators, allowing
fine-tuning of performance and behavior characteristics. Fields not set
in the constructor remain at NCCL’s internal default; values are
validated by the C library when the config is consumed.

#### SEE ALSO
[`ncclConfig_t`](../api/types.md#c.ncclConfig_t) for the description of each field.

#### blocking *: bool | None* *= None*

Blocking (True) or non-blocking (False) communicator behavior. If unset, NCCL uses True.

#### cga_cluster_size *: int | None* *= None*

Cooperative Group Array (CGA) size for kernels (0-8). If unset, NCCL uses 4 for sm90+, 0 otherwise.

#### min_ctas *: int | None* *= None*

Minimum number of CTAs per kernel; positive integer up to 32. If unset, NCCL uses 1.

#### max_ctas *: int | None* *= None*

Maximum number of CTAs per kernel; positive integer up to 32. If unset, NCCL uses 32.

#### net_name *: str | None* *= None*

Network module name (e.g. ‘IB’, ‘Socket’). Case-insensitive. If unset, NCCL auto-selects.

#### split_share *: bool | None* *= None*

Share resources with the child communicator during split. If unset, NCCL uses False.

#### traffic_class *: int | None* *= None*

Traffic class (TC) for network operations (>= 0). Network-specific meaning.

#### comm_name *: str | None* *= None*

User-defined communicator name for logging and profiling.

#### collnet_enable *: bool | None* *= None*

Enable (True) or disable (False) IB SHARP. If unset, NCCL uses False.

#### cta_policy *: [CTAPolicy](#nccl.core.CTAPolicy) | None* *= None*

CTA scheduling policy. If unset, NCCL uses CTAPolicy.DEFAULT.

#### shrink_share *: bool | None* *= None*

Share resources with the child communicator during shrink. If unset, NCCL uses False.

#### nvls_ctas *: int | None* *= None*

Total number of CTAs for NVLS kernels (positive integer). If unset, NCCL auto-determines.

#### n_channels_per_net_peer *: int | None* *= None*

Number of network channels for pairwise communication. Positive integer, rounded up to power of 2. If unset, NCCL uses an AlltoAll-optimized value.

#### nvlink_centric_sched *: bool | None* *= None*

Enable NVLink-centric scheduling. If unset, NCCL uses False.

#### graph_usage_mode *: int | None* *= None*

Graph usage mode (NCCL 2.29+). Supported values are 0 (no graphs), 1 (one graph), 2 (multiple graphs or mix of graph and non-graph). If unset, NCCL uses 2.

#### num_rma_ctx *: int | None* *= None*

Number of RMA contexts (NCCL 2.29+). Positive integer. If unset, NCCL uses 1.

#### max_p2p_peers *: int | None* *= None*

Maximum number of peers any rank will concurrently communicate with using P2P (NCCL 2.30+). Positive integer. If unset, NCCL uses the communicator size.

#### graph_stream_ordering *: int | None* *= None*

Whether NCCL preserves stream-ordering semantics for collectives captured into CUDA graphs. Supported values are 0 (disabled) or 1 (enabled). Cannot be combined with `graph_usage_mode=2`. Also controllable via the `NCCL_GRAPH_STREAM_ORDERING` environment variable. If unset, NCCL uses 1.

## NCCLDevCommRequirements

Used by [`Communicator.create_dev_comm()`](communicator/device_setup.md#nccl.core.Communicator.create_dev_comm). Fields left unset
(`None`) remain at NCCL’s internal default.

### *class* nccl.core.NCCLDevCommRequirements(\*, lsa_multimem: bool | None = None, barrier_count: int | None = None, lsa_barrier_count: int | None = None, rail_gin_barrier_count: int | None = None, lsa_ll_a2a_block_count: int | None = None, lsa_ll_a2a_slot_count: int | None = None, gin_force_enable: bool | None = None, gin_context_count: int | None = None, gin_signal_count: int | None = None, gin_counter_count: int | None = None, gin_connection_type: [NcclGinConnectionType](communicator/device_setup.md#nccl.core.NcclGinConnectionType) | None = None, gin_exclusive_contexts: bool | None = None, gin_queue_depth: int | None = None, world_gin_barrier_count: int | None = None, gin_strong_signals_required: bool | None = None, gin_va_signals_required: bool | None = None)

Bases: `object`

NCCL device communicator requirements configuration.

Provides configuration for device communicator creation, allowing
fine-tuning of resource allocation and device-side communication
behavior. Fields not set in the constructor remain at NCCL’s internal
default; values are validated by the C library when the requirements
are consumed by [`Communicator.create_dev_comm()`](communicator/device_setup.md#nccl.core.Communicator.create_dev_comm).

#### SEE ALSO
[`ncclDevCommRequirements`](../api/device_setup.md#c.ncclDevCommRequirements) for the description of each field.

#### lsa_multimem *: bool | None* *= None*

Enable multimem on the LSA team. If unset, NCCL uses False.

#### barrier_count *: int | None* *= None*

Number of barriers required. If unset, NCCL uses 0.

#### lsa_barrier_count *: int | None* *= None*

Number of LSA barriers. If unset, NCCL uses 0.

#### rail_gin_barrier_count *: int | None* *= None*

Number of railed GIN barriers. If unset, NCCL uses 0.

#### lsa_ll_a2a_block_count *: int | None* *= None*

LSA low-latency all-to-all block count. If unset, NCCL uses 0.

#### lsa_ll_a2a_slot_count *: int | None* *= None*

LSA low-latency all-to-all slot count. If unset, NCCL uses 0.

#### gin_force_enable *: bool | None* *= None*

Force-enable GPU Interconnect Network. If unset, NCCL uses False.

#### gin_context_count *: int | None* *= None*

Number of GIN contexts (hint; actual count may differ). If unset, NCCL uses 4.

#### gin_signal_count *: int | None* *= None*

Number of GIN signals (guaranteed to start at id=0). If unset, NCCL uses 0.

#### gin_counter_count *: int | None* *= None*

Number of GIN counters (guaranteed to start at id=0). If unset, NCCL uses 0.

#### gin_connection_type *: [NcclGinConnectionType](communicator/device_setup.md#nccl.core.NcclGinConnectionType) | None* *= None*

GIN connection type. If unset, NCCL uses NcclGinConnectionType.NONE.

#### gin_exclusive_contexts *: bool | None* *= None*

Use exclusive GIN contexts. If unset, NCCL uses False.

#### gin_queue_depth *: int | None* *= None*

GIN queue depth. If unset, NCCL uses 0.

#### world_gin_barrier_count *: int | None* *= None*

Number of world GIN barriers. If unset, NCCL uses 0.

#### gin_strong_signals_required *: bool | None* *= None*

Whether GIN strong signals are required by kernels using this devComm.
When False, using GIN strong signals results in undefined behavior. If unset, NCCL uses True.

#### gin_va_signals_required *: bool | None* *= None*

Whether GIN VA signals are required by kernels using this devComm.
When False, using GIN VA signals results in undefined behavior. If unset, NCCL uses True.

## CTAPolicy

### *class* nccl.core.CTAPolicy(value, names=<not given>, \*values, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: `IntFlag`

NCCL performance policy for CTA scheduling, used by
[`NCCLConfig.cta_policy`](#nccl.core.NCCLConfig.cta_policy).

#### DEFAULT *= 0*

Default CTA policy.

#### EFFICIENCY *= 1*

Optimize for efficiency.

#### ZERO *= 2*

Zero-CTA optimization.
