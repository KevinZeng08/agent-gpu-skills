# NVSHMEM and the CUDA Model

This section discusses interactions between the CUDA abstract machine model and NVSHMEM.

## The CUDA Execution Model

### Work Submission in CUDA

In the CUDA model, all work (i.e. CUDA tasks) are submitted to the GPU through CUDA streams, which execute their respective tasks in first-in, first-out (FIFO) order. In order to enable concurrency, applications may create multiple CUDA streams and enqueue work on separate streams to indicate that tasks can be processed in parallel. CUDA events can be used to identify dependencies across streams by recording an event on one stream and waiting on the event in another stream.

NVSHMEM stream-based operations (e.g. `nvshmemx_putmem_on_stream`) enqueue the corresponding operation on a CUDA stream and return immediately. The NVSHMEM operation is not performed until it reaches the head of the stream and is executed by the CUDA runtime. All NVSHMEM on-stream operations follow the NVSHMEM memory model. For example, `nvshmemx_quiet_on_stream` operations can be used to order or complete operations, respectively.

On-stream operations also have thread-like semantics because the CUDA layer can execute operations from different streams in parallel with each other and with work performed by the host CPU. However, using stream-based operations doesn’t require NVSHMEM to be initialized with additional threading support. Users must be careful to avoid situations where on-stream operations could violate the threading requirements of a shared object. For example, collective operations that use the same team must not be allowed to execute in parallel. This can happen when a collective operation (e.g. `nvshmem_malloc`) is executed on the CPU in parallel with an operation on a stream that uses the same team (e.g. `nvshmem_barrier_all_on_stream`). In this case, both operations use `NVSHMEM_TEAM_WORLD`. This can also occur when collective operations that use the same team are submitted on different streams without any synchronization to prevent them from executing in parallel.

NVSHMEM’s device-side APIs can be called by a kernel that is executing on the GPU. In the CUDA model, all kernel launches are enqueued on a user-specified stream or on the default stream if none is specified. Thus, usage of NVSHMEM device initiated operations must also take into consideration both the interaction with CUDA stream semantics and with the CUDA thread execution model.

### The CUDA Abstract Machine

The CUDA abstract machine model allows CUDA to insert false dependencies between CUDA tasks in addition to any dependencies explicitly specified by the user (via streams and events). However, these false dependencies added by CUDA must not add cycles to the graph of CUDA tasks. This allows the CUDA layer to manage the execution of tasks and schedule tasks across shared resources. For example, tasks enqueued on CUDA streams are submitted to the GPU through a finite number of hardware streams assigned to the CUDA context. Hardware streams are also processed in FIFO ordering. Thus, when the CUDA layer inserts tasks from several CUDA streams into the same hardware stream, it introduces false dependencies between tasks.

NVSHMEM operations can introduce dependencies between CUDA tasks. For example, kernels can perform point-to-point synchronization and on-stream collectives require participation from multiple PEs. Because these NVSHMEM dependencies are not visible to the CUDA layer when introducing false dependencies, they can lead to cycles in the execution graph and cause deadlock. In the following sections, we highlight several cases in which this can occur and discuss solutions for avoiding deadlock.

## Nonlocal Operations and the CUDA Execution Model

NVSHMEM provides operations that may block until one or more additional operations are performed. We refer to such operations as _nonlocal_.

Examples include NVSHMEM point-to-point synchronization operations, which may block until one or more NVSHMEM operations — performed by either the local PE or a remote PE — update the synchronization variable to satisfy the wait condition. NVSHMEM collective operations may also block until all PEs in the team perform a matching call to the collective operation.

NVSHMEM provides both kernel-initiated and stream-based nonlocal operations. When a stream-based nonlocal NVSHMEM operation is performed, it has the effect of preventing subsequent tasks in the same CUDA stream or in a dependent CUDA stream from being executed (i.e. blocks the CUDA stream) until it has completed. When a kernel-initiated nonlocal NVSHMEM operation is performed it has the effect of holding CUDA execution resources while also blocking the CUDA stream on which it was enqueued.

### CUDA Streams and Circular Dependencies

Users must ensure that NVSHMEM nonlocal operations enqueued on streams don’t form a circular dependence, which can lead to deadlock. For example, consider a situation in which PE 0 enqueues the following operations on a stream (leftmost operation is at the head of the stream):

    PE 0: [ nvshmemx_barrier_all_on_stream, nvshmemx_putmem_signal_on_stream ]

And PE 1 euqueues the following operations on a stream:

    PE 1: [ nvshmemx_signal_wait_until_on_stream, nvshmemx_barrier_all_on_stream ]

The signal wait operation at PE 1 would be satisfied by the putmem-with-signal operation enqueued at PE 0. However, PE 0 is blocked in the barrier operation, preventing it from executing the putmem-with-signal operation. Neither PE can make forward progress, resulting in deadlock.

### CUDA Stream Order and Execution Resources

Similarly, consider a message exchange in which a kernel named notify_kernel calls `nvshmem_putmem_signal` to send a message to a peer PE and wait_kernel performs a corresponding `nvshmem_signal_wait_until` operation to wait for a message from a peer PE. These operations could be enqueued into separate streams as follows at PE 0 and PE 1:

    PE 0, Stream A: [ wait_kernel ]
    PE 0, Stream B: [ notify_kernel ]
    PE 1, Stream A: [ wait_kernel ]
    PE 1, Stream B: [ notify_kernel ]

The user has enqueued these operations into separate streams because the can be executed in parallel. Because the kernels are enqueued in separate streams, the CUDA runtime can execute them in any order. If the CUDA runtime executes the wait_kernel first, it will block and hold CUDA execution resources. These resources may be needed to execute the notify_kernel, thus preventing it from executing. In addition, even when sufficient resources are available, the CUDA runtime may insert a device synchronization between kernels that prevent them from running in parallel. If both PEs execute the wait_kernel first, it can result in a deadlock. The deadlock can be prevented by enqueueing both kernels in the same stream with the notify kernel first or by inserting a CUDA event to prevent the wait_kernel from being executed first.

### CUDA Streams and False Circular Dependencies

The CUDA model allows users to create any number of streams to describe their workload. The CUDA runtime pushes work from these streams into GPU work queues that are managed by the GPU. Work in GPU work queues is processed in FIFO order. The CUDA runtime respects any stream-order and CUDA event dependencies when assigning work from streams to GPU work queues. NVSHMEM’s nonlocal dependencies are not visible to the CUDA runtime and users must be careful to introduce a CUDA-visible dependence where needed to prevent the CUDA layer from serializing tasks in an order that can lead to deadlock.

For example, consider the message exchange example given above:

    PE 0, Stream A: [ wait_kernel ] PE 0, Stream B: [ notify_kernel ]
    PE 1, Stream A: [ wait_kernel ] PE 1, Stream B: [ notify_kernel ]

The CUDA runtime may serialize these tasks as follows, which will cause deadlock:

    PE 0, GPU Work Queue: [ wait_kernel, notify_kernel ]
    PE 1, GPU Work Queue: [ wait_kernel, notify_kernel ]

Users can introduce a dependency by enqueueing notify_kernel and wait_kernel on the same stream or by introducing a CUDA event as follows:

    PE 0, Stream A: [ cudaEventSynchronize(e), wait_kernel ]
    PE 0, Stream B: [ notify_kernel, cudaEventRecord(e) ]
    PE 1, Stream A: [ cudaEventSynchronize(e), wait_kernel ]
    PE 1, Stream B: [ notify_kernel, cudaEventRecord(e) ]

### Intra-Kernel Synchronization

The CUDA throughput computing model allows users to specify grid and thread block dimensions much larger than what can be executed in parallel on a given GPU. Exposing such large volumes of work can allow the GPU to process the work efficiently. However, CUDA does not guarantee preemptive thread scheduling. Thus, when a thread in a kernel performs a nonlocal operation, it can block execution of other threads in the same kernel. This, in turn, can prevent the thread that would make the matching call from executing leading to a deadlock. This issue can be avoided by launching kernels using the `cudaLaunchCooperativeKernel` API, which ensures that threads in the kernel can safely synchronize with one another without causing deadlock.

### Ensuring Safe Nonlocal Operations Using NVSHMEM Cooperative Kernel Launch

To simplify the use of nonlocal NVSHMEM functions in CUDA kernels, NVSHMEM provides Kernel Launch Routines. The `nvshmemx_collective_launch` function can be used to launch CUDA kernels on the GPU when the CUDA kernels use NVSHMEM synchronization or collective APIs (e.g., `nvshmem_wait`, `nvshmem_barrier`, `nvshmem_barrier_all`, or any other collective operation). CUDA kernels that do not use synchronizing NVSHMEM APIs or that do not use NVSHMEM APIs at all, are not required to be launched by this API.

This call is collective across the PEs in the NVSHMEM job. It ensures both that the kernel fits in the GPU at each PE and that the kernels are launched simultaneously across all PEs.

## Implicitly Asynchronous cudaMemcpy

The `cudaMemcpy` and `cudaMemset` routines can exhibit asynchronous behavior as described here. When performed asynchronously, these operations may not have completed before any subsequent NVSHMEM operations are performed. When not using VMM, NVSHMEM sets the `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS` flag on symmetric memory, forcing these operations to be performed synchronously. However, this flag is not currently supported on VMM allocations. To avoid a possible data race, users can use the explicitly asynchronous `cudaMemcpyAsync` and `cudaMemsetAsync` operations and synchronize the corresponding stream.

For example, the following host code contains a possible race when `cudaMemcpy` can be performed asynchronously.

    cudaMemcpy(out_d, &out, sizeof(unsigned int), cudaMemcpyHostToDevice);
    nvshmem_uint_or_reduce(NVSHMEM_TEAM_WORLD, final_out_d, out_d, 1);

The race can be removed using `cudaMemcpyAsync`.

    cudaMemcpyAsync(out_d, &out, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    nvshmem_uint_or_reduce(NVSHMEM_TEAM_WORLD, final_out_d, out_d, 1);
