# Introduction

NVSHMEM is a software library that implements the OpenSHMEM application programming interface (API) for clusters of NVIDIA ® GPUs. OpenSHMEM is a community standard, one-sided communication API that provides a partitioned global address space (PGAS) parallel programming model. A key goal of the OpenSHMEM specification, and also of NVSHMEM, is to provide an interface that is convenient to use, while also providing high performance with minimal software overheads.

The OpenSHMEM specification is under active development with regular releases that expand its feature set and extend its ability to use emerging node and cluster architectures. The current version of NVSHMEM is based on the OpenSHMEM version 1.3 APIs and also includes many features from later versions of OpenSHMEM. Although NVSHMEM is based on OpenSHMEM, there are important differences that are detailed in this guide.

NVSHMEM provides an easy-to-use host-side interface to allocate symmetric memory, which can be distributed across a cluster of NVIDIA GPUs that are interconnected with NVLink, PCIe, and InfiniBand. Device-side APIs can be called by CUDA kernel threads to efficiently access locations in symmetric memory through one-sided read (get), write (put), and atomic update API calls. In addition, symmetric memory that is directly accessible to a GPU, for example, a memory region that is located on the local GPU or a peer GPU connected via NVLink, can be queried and accessed directly via a pointer that is provided by the NVSHMEM library.

## Project Homepage

The project homepage provides links to the latest release, documentation, and source code.

## Key Features

NVSHMEM extends the OpenSHMEM APIs to support clusters of NVIDIA GPUs. The following provides information about some key extensions:

>   * Support for the symmetric allocation of GPU memory.
>   * Support for GPU-initiated communication, including support for CUDA types.
>   * A new API call to collectively launch CUDA kernels across a set of GPUs.
>   * Stream-based APIs that allow data movement operations that were initiated from the CPU to be offloaded to the GPU and ordered with regard to a CUDA stream.
>   * Threadgroup communication where threads from entire warps or thread blocks in a CUDA kernel can collectively participate in a communication operation.
>   * Differentiation between synchronizing and non-synchronizing operations to benefit from strengths, weak or strong, of the operations in the GPU memory model.
>

Here is a summary of the differences between NVSHMEM and OpenSHMEM.

>   * API names are prefixed with “nv” to enable hybrid usage of NVSHMEM with an existing OpenSHMEM library.
>   * All buffer arguments to NVSHMEM communication routines must be symmetric.
>   * NVSHMEM provides weak ordering for data that is is returned by blocking operations that fetch data. Ordering can be enforced via the `nvshmem_fence` operation.
>

Refer to openshmem.org for more information about the OpenSHMEM specification.

## Communication Transports

In addition to local GPU-GPU communication, NVSHMEM also supports one-sided network communication through two additional channels:

  1. InfiniBand or RoCE over the verbs interface
  2. InfiniBand or RoCE using UCX (Experimental).

The compilation and use of these transports can be modified by environment variables. For more information on compiling NVSHMEM, refer to the installation guide For runtime environment configuration, refer to the Environment Variable section.

## Advantages Of NVSHMEM

NVSHMEM aggregates the memory of multiple GPUs in a cluster into a PGAS that enables fine-grained GPU-to-GPU data movement and synchronization from within a CUDA kernel.

Using NVSHMEM, developers can write long running kernels that include communication and computation, which reduces the need for synchronization with the CPU. These composite kernels also allow for a fine-grain overlap of computation with communication as a result of thread warp scheduling on the GPU. NVSHMEM GPU-initiated communication reduces overhead that results from kernel launches, calls to CUDA API, and CPU-GPU synchronization. Reducing these overheads can enable significant gains in strong scaling of application workloads.

**Note** : When necessary, NVSHMEM also provides the flexibility of CPU-side calls for inter-GPU communication outside of CUDA kernels.

The Message Passing Interface (MPI) is one of the most commonly used communication libraries for scalable, high performance computing. A key design principle of NVSHMEM is to support fine-grain, highly-concurrent GPU-to-GPU communication from CUDA kernels. Using conventional MPI send and receive operations in such a communication regime can lead to significant efficiency challenges. Because of the need to match send and receive operations, the following issues can occur:

>   * MPI implementations can incur high locking (or atomics) overheads for shared data structures that are involved in messaging.
>   * Serialization overheads that result from the MPI message ordering requirements.
>   * Protocol overheads that result from messages arriving at the receiver before they have posted the corresponding receive operation.
>

Although there have been efforts to reduce the overhead of critical sections by using fine-grained locking and multiple network end-points per process, matching between send and receive operations is inherent to the send-receive communication model in MPI and is challenging to scale to highly threaded environments that are presented by GPUs. In contrast, one-sided communication primitives avoid these bottlenecks by enabling the initiating thread to specify all the information that is required to complete a data transfer. One-sided operations can be directly translated to RDMA primitives that are exposed by the network hardware or to load/store operations on the NVLink fabric. Using asynchronous APIs, one-sided primitives also make it programmatically easier to interleave computation and communication, which has the potential for better overlap.

## GPU-Initiated Communication And Strong Scaling

NVSHMEM support for GPU-initiated communication can significantly reduce communication and synchronization overheads, which leads to strong scaling improvements.

Strong scaling refers to how the solution time of a fixed problem changes as the number of processors is increased. It is the ability to solve a fixed problem faster by increasing the number of processors. This is a critical metric for many scientific, engineering, and data analytics applications as increasingly large clusters and more powerful GPUs become available.

Current state-of-the-art applications that run on GPU clusters typically offload computation phases onto the GPU and rely on the CPU to manage communication between cluster nodes, by using MPI or OpenSHMEM. Depending on the CPU for communication limits strong scalability because of the overhead of repeated kernel launches, CPU-GPU synchronization, underutilization of the GPU during communication phases, and underutilization of the network during compute phases. Some of these issues can be addressed by restructuring the application code to overlap independent compute and communication phases using CUDA streams. These optimizations can lead to complex application code and the benefits usually diminish as the problem size per GPU becomes smaller.

As a problem is strong-scaled, the Amdahl’s fraction of the execution that corresponds to CPU-GPU synchronization and communication increases. As a result, minimizing these overheads is critical for the strong scaling of applications on GPU clusters. GPUs are also designed for throughput and have enough state and parallelism to hide long latencies to global memory, which allows GPUs to be efficient at hiding data movement overheads. Following the CUDA programming model and best practices and using NVSHMEM for GPU-initiated communication allows developers to take advantage of these latency hiding capabilities.
