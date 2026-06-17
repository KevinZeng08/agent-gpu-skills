# Device API

The Device API allows communication to be initiated and performed from device (GPU) code. It is organized into the
following areas:

* **Host-Side Setup** — Creating and configuring device communicators, querying properties, host-accessible device
  pointer functions, and related types.
* **Memory and LSA** — Load/store accessible (LSA) memory, barriers, pointer accessors, and multimem.
* **GIN (GPU-Initiated Networking)** — One-sided transfers, signals, counters, and network barriers.
* **Reduce, Broadcast, and Fused Building Blocks** — Building blocks for computation-fused kernels: reduce, copy
  (broadcast), and reduce-then-copy; used to implement algorithms such as AllReduce, AllGather, and ReduceScatter.

For an introduction and usage examples, see [Device-Initiated Communication](../usage/deviceapi.md).

* [Device API – Host-Side Setup](device_setup.md)
  * [Host-Side Setup](device_setup.md#host-side-setup)
  * [Host-Accessible Device Pointer Functions](device_setup.md#host-accessible-device-pointer-functions)
* [Device API – Memory and LSA](device_memory.md)
  * [LSA](device_memory.md#lsa)
  * [Multimem](device_memory.md#multimem)
* [Device API – GIN](device_gin.md)
  * [GIN](device_gin.md#gin)
* [Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels](device_reducecopy.md)
  * [Compile-Time Requirements](device_reducecopy.md#compile-time-requirements)
  * [API Overview](device_reducecopy.md#api-overview)
  * [ReduceSum — N Sources to One Destination](device_reducecopy.md#reducesum-n-sources-to-one-destination)
  * [Copy (Broadcast) — One Source to N Destinations](device_reducecopy.md#copy-broadcast-one-source-to-n-destinations)
  * [ReduceSumCopy](device_reducecopy.md#reducesumcopy)
  * [Lambda-Based (Custom Layouts)](device_reducecopy.md#lambda-based-custom-layouts)
  * [Custom Reduction Operators](device_reducecopy.md#custom-reduction-operators)
