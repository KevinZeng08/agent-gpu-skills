# Communicator

The [`Communicator`](communicator/class.md#nccl.core.Communicator) class and its methods, organized by lifecycle
stage and operation kind:

- [Communicator Class](communicator/class.md) — the class itself, its constructor, and
  per-instance properties for identity and device-API capability.
- [Creation and Lifecycle Methods](communicator/lifecycle.md) — creating, splitting, growing, and tearing
  down communicators.
- [Collective Communication Methods](communicator/collectives.md) — collective communication methods
  (allreduce, broadcast, gather, …).
- [Point-to-Point and Signal Methods](communicator/p2p.md) — point-to-point and signal methods
  (send / recv / signal / wait_signal / put_signal).
- [Memory Registration Methods](communicator/registration.md) — buffer and window registration for
  zero-copy and RMA.
- [Device Communicator Setup](communicator/device_setup.md) — host-side bootstrap of a device
  communicator.
- [Status and Utility Methods](communicator/status.md) — error queries and resource cleanup.

* [Communicator Class](communicator/class.md)
  * [`Communicator`](communicator/class.md#nccl.core.Communicator)
  * [Properties](communicator/class.md#properties)
* [Creation and Lifecycle Methods](communicator/lifecycle.md)
  * [Construction](communicator/lifecycle.md#construction)
  * [Bootstrap identifier](communicator/lifecycle.md#bootstrap-identifier)
  * [Splitting and growing](communicator/lifecycle.md#splitting-and-growing)
  * [Teardown](communicator/lifecycle.md#teardown)
  * [Pause and resume](communicator/lifecycle.md#pause-and-resume)
  * [Flag enums](communicator/lifecycle.md#flag-enums)
* [Collective Communication Methods](communicator/collectives.md)
  * [allreduce](communicator/collectives.md#allreduce)
  * [broadcast](communicator/collectives.md#broadcast)
  * [reduce](communicator/collectives.md#reduce)
  * [allgather](communicator/collectives.md#allgather)
  * [reduce_scatter](communicator/collectives.md#reduce-scatter)
  * [alltoall](communicator/collectives.md#alltoall)
  * [gather](communicator/collectives.md#gather)
  * [scatter](communicator/collectives.md#scatter)
  * [create_pre_mul_sum](communicator/collectives.md#create-pre-mul-sum)
* [Point-to-Point and Signal Methods](communicator/p2p.md)
  * [send](communicator/p2p.md#send)
  * [recv](communicator/p2p.md#recv)
  * [signal](communicator/p2p.md#signal)
  * [wait_signal](communicator/p2p.md#wait-signal)
  * [put_signal](communicator/p2p.md#put-signal)
  * [WaitSignalDesc](communicator/p2p.md#waitsignaldesc)
* [Memory Registration Methods](communicator/registration.md)
  * [register_buffer](communicator/registration.md#register-buffer)
  * [register_window](communicator/registration.md#register-window)
  * [WindowFlag](communicator/registration.md#windowflag)
* [Device Communicator Setup](communicator/device_setup.md)
  * [create_dev_comm](communicator/device_setup.md#create-dev-comm)
  * [GIN type enums](communicator/device_setup.md#gin-type-enums)
* [Status and Utility Methods](communicator/status.md)
  * [close_all_resources](communicator/status.md#close-all-resources)
  * [get_last_error](communicator/status.md#get-last-error)
  * [get_async_error](communicator/status.md#get-async-error)
  * [get_mem_stat](communicator/status.md#get-mem-stat)
  * [NcclCommMemStat](communicator/status.md#ncclcommmemstat)
  * [get_error_string](communicator/status.md#get-error-string)
