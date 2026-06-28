# Troubleshooting And FAQs

## General FAQs

_Q: What does the following runtime warning imply?_

    WARN: IB HCA and GPU are not connected to a PCIe switch so InfiniBand
    performance can be limited depending on the CPU generation

A: This warning is related to the HCA to GPU mapping of the platform. For more information, refer to the `SHMEM_HCA_PE_MAPPING` variable in Environment Variables.

_Q: What does the following runtime error indicate?_

    src/bootstrap/bootstrap_loader.cpp:46: NULL value Bootstrap unable to load 'nvshmem_bootstrap_mpi.so'
        nvshmem_bootstrap_mpi.so: cannot open shared object file: No such file or directory
    src/bootstrap/bootstrap.cpp:26: non-zero status: -1 bootstrap_loader_init returned error
    src/init/init.cpp:101: non-zero status: 7 bootstrap_init failed
    src/init/init.cpp:767: non-zero status: 7 nvshmem_bootstrap failed
    src/init/init.cpp:792: non-zero status: 7: Success, exiting... aborting due to error in nvshmemi_init_thread

A: NVSHMEM uses dynamically loaded bootstrap modules for several bootstraps, including MPI, OpenSHMEM, and PMIx. The above error indicates that the bootstrap module for MPI could not be loaded. Ensure that the NVSHMEM library directory is in the system search path for the dynamic linker or that the `LD_LIBRARY_PATH` variable includes the NVSHMEM library directory. Alternatively, you can set the `NSHMEM_BOOTSTRAP_PLUGIN` environment variable to help NVSHMEM locate the plugin. If the plugins were not built during the NVSHMEM installation process or if you need support for a different library (e.g. MPI library) than was used during installation, the NVSHMEM bootstrap plugins can be built from the source code provided under the `share/nvshmem` directory.

_Q: What does the following runtime error indicate?_

    src/comm/transports/ibrc/ibrc.cpp:: NULL value mem registration failed.

A: This occurs if GPUDirect RDMA is not enabled on the platform, thereby preventing registration of `cudaMalloc` memory with the InfiniBand driver. This usually indicates that the `nv_peer_mem` kernel module is absent. When `nv_peer_mem` is installed, output from `lsmod` is similar to the following:

    ~$ lsmod | grep nv_peer_mem
    nv_peer_mem               20480 0
    ib_core                   241664 11
    rdma_cm,ib_cm,iw_cm,nv_peer_mem,mlx4_ib,mlx5_ib,ib_ucm,ib_umad,ib_uverbs,rdma_ucm,ib_ipoib
    nvidia                  17596416 226
    nv_peer_mem,gdrdrv,nvidia_modeset,nvidia_uvm

nv_peer_mem is available here.

_Q: What does the following runtime error indicate?_

    src/comm/transports/ibrc/ibrc.cpp: NULL value get_device_list failed

A: This occurs when ibverbs library is present on the system but the library is not able to detect any InfiniBand devices on the system. Make sure that the InfiniBand devices are available and/or are in a working state.

_Q: What does the following runtime error indicate?_

    src/comm/transports/libfabric/libfabric.cpp:482: non-zero status: -12 Error registering memory region: Cannot allocate memory
    src/mem/mem.cpp:464: non-zero status: 7 transport get memhandle failed
    src/comm/device/proxy_device.cu:695: NULL value failed allocating proxy_channel_g_bufchannel creation failed

A: This can occur when memory allocated through the CUDA VMM interfaces is not supported by the networking stack. You can try setting NVSHMEM_DISABLE_CUDA_VMM=1 in the environment to disable the use of CUDA VMM interfaces when NVSHMEM allocates memory.

_Q: What would cause runtime errors relating to ibv_poll_cq?_

For example:

    src/comm/transports/ibrc/ibrc.cpp:962: non-zero status: 10 ibv_poll_cq failed, status: 10
    src/comm/transports/ibrc/ibrc.cpp:1021: non-zero status: 7 progress_send failed, outstanding_count: 0
    src/comm/transports/ibrc/ibrc.cpp:1294: non-zero status: 7 check_poll failed
    src/comm/proxy/proxy.cu:progress_quiet:612: aborting due to error in progress_quiet

A: These types of errors occur in the NVSHMEM ibrc remote transport in response to infiniband work request failures. Work requests can fail for many reasons as documented in the ibv_poll_cq man page. In NVSHMEM, we post the code for the ibv_poll_cq error in the line “ibv_poll_cq failed, status: 10”. That status value can be looked up in the ibv_poll_cq docs to determine the cause of the failure.

In NVSHMEM, the most common reasons for IB failures are either status: 10 - Remote Protection Error or status: 4 - Local Protection Error. The first error happens when an address not on the NVSHMEM symmetric heap is used as the remote buffer in an RMA or atomic operation. The second error happens when the local buffer was neither in the symmetric heap nor registered with NVSHMEM as a local buffer. Both errors can also happen if an address returned from nvshmem_ptr or nvshmemx_mc_ptr is used in an NVSHMEM RMA/AMO operation.

_Q: My application uses the CMake build system. Adding NVSHMEM to the build system breaks for a CMake version below 3.11. Why?_

A: Device linking support was added in version 3.11 which NVSHMEM requires.

_Q: Why does a CMake build of my NVSHMEM application fail with version 3.12 but does not with an earlier version?_

A: A new CMake policy adds `-pthread` to the `nvcc` device linking causing the linking failure. Before 3.12, the default policy did not add `-pthread`. For 3.12 and newer, add `cmake_policy(SET CMP0074 OLD)` to `CMakeLists.txt`.

_Q: What CMake settings needed to build CUDA or NVSHMEM applications?_

A: Add the following to the CMake file, substituting the target GPU architecture in place of `compute_70` and `sm_70`.

    string(APPEND CMAKE_CUDA_FLAGS "-gencode arch=compute_70,code=sm_70")

_Q: Why does my NVSHMEM Hydra job become non-responsive on Summit?_

A: Summit requires the additional option `--launcher ssh` to be passed to `nvshmrun` at the command-line.

_Q: Can multiple PEs share the same GPU with NVSHMEM?_

A: Until NVSHMEM 2.2.1 NVSHMEM assumes a 1:1 mapping of PEs to GPUs. Jobs launched with more PEs than the available GPUs were not supported. Since NVSHMEM 2.4.1, limited support for Multiple Processes per GPU (GPU) is available. More details about this can be found in Multiprocess GPU Support.

_Q. What is the right way to use CUDA_VISIBLE_DEVICES with NVSHMEM?_

A. When using `CUDA_VISIBLE_DEVICES` with NVSHMEM, all PEs should be passed the same value of `CUDA_VISIBLE_DEVICES`. Note that we may change this in a future NVSHMEM version.

## Prerequisite FAQs

_Q: Does NVSHMEM require CUDA?_

A: Yes. CUDA must be installed to use NVSHMEM. Refer to the installation guide for version requirements. NVSHMEM is a communication library intended to be used for efficient data movement and synchronization between two or more GPUs. It is currently not intended for data movement that does not involve GPUs.

_Q: Does NVSHMEM require MPI?_

A: No. NVSHMEM applications without MPI dependencies can use NVSHMEM and be launched with the Hydra launcher. Hydra can be installed using the installation script included with NVSHMEM. This script installs the Hydra launcher binaries with the names `nvshmrun` and `nvshmrun.hydra`. An externally installed copy of the Hydra launcher can also be used, which typically installs the Hydra launcher binaries with the names `mpiexec`, `mpirun`, and `mpiexec.hydra`.

_Q: My NVSHMEM job runs on NVIDIA Volta GPUs but hangs on NVIDIA Kepler GPUs. Why does this happen?_

A: NVSHMEM Synchronizing APIs inside the CUDA kernel are only supported on NVIDIA Volta and newer GPUs.

_Q: What does the following runtime error indicate when installing NVSHMEM4Py?_

      building 'nvshmem.bindings.nvshmem' extension
      creating build/temp.linux-x86_64-cpython-310/nvshmem/bindings
      x86_64-linux-gnu-g++ -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -fPIC -I/home/benjaming/repo/nvshmem_venv/include -I/usr/include/python3.10 -c nvshmem/bindings/nvshmem.cpp -o build/temp.linux-x86_64-cpython-310/nvshmem/bindings/nvshmem.o
      nvshmem/bindings/nvshmem.cpp:1137:10: fatal error: cuda_runtime_api.h: No such file or directory
      1137 | #include "cuda_runtime_api.h"
          |          ^~~~~~~~~~~~~~~~~~~~
      compilation terminated.
      error: command '/bin/x86_64-linux-gnu-g++' failed with exit code 1
      [end of output]

    note: This error originates from a subprocess, and is likely not a problem with pip.
    ERROR: Failed building wheel for nvshmem4py-cu12
    Failed to build nvshmem4py-cu12
    ERROR: Could not build wheels for nvshmem4py-cu12, which is required to install pyproject.toml-based projects

A: NVSHMEM4Py’s internal bindings require the CUDA runtime API headers to be available on your compiler’s include path. This error means the compiler cannot find the CUDA runtime headers. To resolve this, you should make sure that the CUDA toolkit’s `include` directory is in your compiler’s include path. For example, setting `CPPFLAGS="-I<CUDA_INSTALL_PATH>/include"`, where `CUDA_INSTALL_PATH` is the place you have the CUDA toolkit installed will ensure the CUDA runtime headers are found. Alternaively, you can set the `CPATH` environment variable to include the CUDA toolkit’s `include` directory.

## Running NVSHMEM Programs FAQs

_Q: I get a missing symbol error when I try to launch an NVSHMEM application using PMIx with the OpenMPI implementation of mpirun. How can I fix this?_

A: There is a known incompatibility between the internal PMIx implementation in OpenMPI and the one we use in NVSHMEM. Trying to run an nvshmem application with OpenMPI using the internal PMIx will most likely result in the following error:

    pmix_mca_base_component_repository_open: unable to open mca_gds_ds21:
    perhaps a missing symbol, or compiled for a different version of OpenPMIx (ignored)

This can be worked around by compiling OpenMPI with an external implementation of PMIx using the `--with-pmix={PATH_TO_PMIX}` configure option.

## Interoperability With MPI FAQs

_Q: Can NVSHMEM be used in MPI applications?_

A: Yes. NVSHMEM provides an initialization API that takes an MPI communicator as an attribute. Each MPI rank in the communicator becomes an OpenSHMEM PE. Currently, NVSHMEM has been tested with OpenMPI 4.0.0. In principle, other OpenMPI derivatives such as SpectrumMPI (available on Summit and Sierra) are also expected to work.

_Q: Passing NVSHMEM allocated buffers to MPI results in an error. What is the resolution?_

NVSHMEM symmetric memory is mapped using CUDA IPC. When these buffers are passed to MPI, MPI may try to map these buffers for CUDA IPC a second time. Prior to CUDA 11.2 this would result in an error. It is now supported in CUDA 11.2 and later.

As of NVSHMEM v2.2.1, symmetric memory is mapped using the CUDA VMM APIs. This method of mapping the symmetric heap is not compatible with some CUDA-aware MPI libraries, resulting in a crash like the following:

    cma_ep.c:97   process_vm_readv(pid=434391 length=524288) returned -1: Bad address
    ==== backtrace (tid: 434392) ====
    0 0x0000000000002558 uct_cma_ep_tx()  src/uct/sm/scopy/cma/cma_ep.c:95
    1 0x000000000001a17c uct_scopy_ep_progress_tx()  src/uct/sm/scopy/base/scopy_ep.c:151
    2 0x000000000004f7c1 ucs_arbiter_dispatch_nonempty()  src/ucs/datastruct/arbiter.c:321
    3 0x0000000000019ce4 ucs_arbiter_dispatch()  src/ucs/datastruct/arbiter.h:386
    4 0x000000000004fcfe ucs_callbackq_slow_proxy()  src/ucs/datastruct/callbackq.c:402
    5 0x0000000000034eba ucs_callbackq_dispatch()  src/ucs/datastruct/callbackq.h:211
    6 0x0000000000034eba uct_worker_progress()  src/uct/api/uct.h:2592
    7 0x0000000000034eba ucp_worker_progress()  src/ucp/core/ucp_worker.c:2635
    8 0x000000000003694c opal_progress()  ../opal/runtime/opal_progress.c:231
    9 0x000000000004c983 ompi_request_default_test()  ../ompi/request/req_test.c:96
    10 0x000000000004c983 ompi_request_default_test()  ../ompi/request/req_test.c:42
    11 0x0000000000071d35 PMPI_Test()  ../ompi/mpi/c/profile/ptest.c:65
    ...

These issues can be avoided by setting `NVSHMEM_DISABLE_CUDA_VMM=1`.

## Interoperability With OpenSHMEM FAQs

_Q: Can NVSHMEM be used in OpenSHMEM applications?_

A: Yes. NVSHMEM provides an initialization API that supports running NVSHMEM on top of an OpenMPI/OSHMEM job. Each OSHMEM PE maps 1:1 to an NVSHMEM PE. NVSHMEM has been tested with OpenMPI 4.0.0/OSHMEM and OpenMPI3+/OSHMEM depends on UCX (NVSHMEM has been tested with UCX 1.4.0). The OpenMPI-4.0.0 installation must be configured with the `--with-ucx` flag to enable OpenSHMEM + NVSHMEM interoperability.

_Q: How do I initialize NVSHMEM when using Open MPI SHMEM (OSHMEM)?_

A: OSHMEM has a known issue that it erroneously sets the CUDA context on device 0 during shmem_init(). If after calling shmem_init(), NVSHMEM is initialized, NVSHMEM detects that the device has been set to device 0 for all the PEs and hence initializes itself in multiprocess GPU sharing mode with all PEs assigned to GPU 0. Therefore, when running with OSHMEM user must set the desired CUDA context before calling NVSHMEM initialization.

## GPU-GPU Interconnection FAQs

_Q: Can I use NVSHMEM to transfer data across GPUs on different sockets?_

A: Yes, if there is an InfiniBand NIC accessible to GPUs on both the sockets. Otherwise, NVSHMEM requires that all GPUs are P2P-accessible.

_Q: Can I use NVSHMEM to transfer data between P2P-accessible GPUs that are connected by PCIe?_

A: Yes, NVSHMEM supports PCIe. However, when using PCIe for P2P communication, either InfiniBand support is required to use NVSHMEM atomic memory operations API or one has to use NVSHMEM’s UCX transport (that will use sockets for atomics when IB is absent).

_Q: Can I use NVSHMEM to transfer data between GPUs on different hosts connected by InfiniBand?_

A: Yes. NVSHMEM supports InfiniBand. Strided-RMA (shmem_iput/iget) operations are not supported over InfiniBand.

_Q: Can I run NVSHMEM on a host without InfiniBand NICs?_

A: Yes. Support on P2P platforms remains unchanged.

_Q: Can I run NVSHMEM on a host with InfiniBand NICs where some NICs are disabled or configured in a non-InfiniBand mode?_

A: Yes. See the Useful Environment Variables section for how to explicitly specify NIC affinity to PEs.

_Q: How do I check if a given remote PE is P2P-accessible?_

A: The function `nvshmem_ptr` returns a non-null pointer if the remote PE is P2P-accessible. Alternatively, the team `NVSHMEM_TEAM_SHARED` consists of all PEs that are P2P-accessible, so use `nvshmem_team_translate_pe` to check if a given PE is in this team, for example:

    int is_p2p_accessible = nvshmem_team_translate_pe(NVSHMEM_TEAM_WORLD,
                                                      remote_pe,
                                                      NVSHMEM_TEAM_SHARED) != -1;

## NVSHMEM API Usage FAQs

_Q: What’s the difference between, say, nvshmemx_putmem_on_stream and nvshmemx_putmem_nbi_on_stream? It seems both are asynchronous to the host thread and ordered with respect to a given stream._

A: The function `nvshmemx_putmem_nbi_on_stream` is implemented in a more deferred way by not issuing the transfer immediately but making it wait on an event at the end of the stream. If there is another transfer in process at the same time (on another stream), bandwidth could be shared. If the application can avoid this, `nvshmemx_putmem_nbi_on_stream` gives the flexibility to express this intent to NVSHMEM. But NVSHMEM currently does not track activity on all CUDA streams. The current implementation records an event on the user provided stream, makes an NVSHMEM internal stream wait on the event, and then issues a put on the internal stream. If all nbi puts land on the same internal stream, they are serialized so that the bandwidth is used exclusively.

_Q: Can I issue multiple nvshmemx_barrier_all_on_stream on multiple streams concurrently and then cudaStreamSynchronize on each stream?_

A: Multiple concurrent `nvshmemx_barrier_all_on_stream` / `nvshmem_barrier_all` calls are not valid. Only one barrier (or any other collective) among the same set of PEs can be in-flight at any given time. To use concurrent barriers among partially overlapping teams, `syncneighborhood_kernel` can be used as a template to implement a custom barrier. See the following for an example of a custom barrier (multi-gpu-programming-models).

_Q: Suppose there are in-flight nvshmem_putmem_on_stream operations. Does nvshmem_barrier_all() ensure completion of the pending NVSHMEM operations on streams?_

A: The `nvshmem_barrier_all()` operation does not ensure completion of the pending NVSHMEM operations on streams. The `cudaStreamSynchronize` function should be called before calling `nvshmem_barrier_all`.

_Q: Why is nvshmem_quiet necessary in the syncneighborhood_kernel?_

A: It is required by `nvshmem_barrier` semantics. As stated in multi-gpu-programming-models, “nvshmem_barrier ensures that all previously issued stores and remote memory updates, including AMO and RMA operations, done by any of the PEs in the team on the default context are complete before returning.”

_Q: If a kernel uses nvshmem_put_block instead of nvshmem_p, is nvshmem_quiet still required?_

A: It is required per OpenSHMEM’s requirement to put semantics which do not guarantee delivery of data to the destination array on the remote PE. For more information, see multi-gpu-programming-models.

_Q: I use the host-side blocking API, nvshmem_putmem_on_stream, on the same CUDA stream that I want to be delivered at the target in order. Is nvshmem_quiet required even though there is no non-blocking call and they are issued in separate kernels?_

A: In the current implementation, `nvshmem_putmem_on_stream` includes quiet. However, it is only required to release the local buffer and not necessarily deliver at the target by the OpenSHMEM spec.

_Q: Is it sufficient to use a nvshmem_fence (instead of a nvshmem_quiet) in the above case if the target is the same PE?_

A: In the current implementation, all messages to the same PE are delivered in the order they are received by the HCA, which follows the stream order. So, even `nvshmem_fence` is not required. These are not the semantics provided by the OpenSHMEM specification, however. The `nvshmem_putmem_on_stream` function on the same CUDA stream only ensures that the local buffers for the transfers will be released in the same order.

_Q: When nvshmem_quiet is used inside a device kernel, is the quiet operation scoped within the stream the kernel is running on? In other words, does it ensure completion of all operations or only those issued to the same stream?_

A: It ensures completion of all operations that are GPU-initiated. A `nvshmem_quiet` call on the device does not quiet in-flight operations from the host.

## Debugging FAQs

NOTE: Ensure you follow the CUDA Best Practices Guide to ease debugging CUDA programs. For example, read Error Handling.

_Q: Is there any hint to diagnose the hang?_

A: Check if there are stream 0 blocking CUDA calls from the application, like `cudaDeviceSynchronize` or `cudaMemcpy`, especially in the iterative phase of the application. Stream 0 blocking calls in the initialization and finalization phases are usually safe. Check if the priority of the user stream used for NVSHMEM `_on_stream` calls is explicitly set with `cudaStreamCreateWithPriority`. Check that the determinism of the hang changes with single-node (all pairs of GPUs connected by NVLink or PCI-E only) compared to single-node (GPUs on different sockets connected by InfiniBand loopback) or multi-node (GPUs connected by InfiniBand).

_Q: How do I dump debugging information?_

A: Refer to the runtime environment variables: `NVSHMEM_INFO`, `NVSHMEM_DEBUG`, and `NVSHMEM_DEBUG_FILE` in Environment Variables.

_Q: Why is the receive buffer not updated with remote data even after synchronization with a flag?_

A: For synchronization with flag, the application must use `nvshmem_wait_until` or `nvshmem_test` API. A plain while loop or if condition to check flag value is not sufficient. NVSHMEM needs to perform consistency operation to ensure that the data is visible to the GPU after synchronization using flag value.

## Miscellaneous FAQs

_Q: Does pointer arithmetic work with shmem pointers? For example,_

    int* outmsg = (int *) shmem_malloc(2* sizeof(int));
    shmem_int_p(target + 1, mype, peer);

A: Yes.

_Q: Can I avoid cudaDeviceSynchronize + MPI_Barrier to synchronize across multiple GPUs?_

A: Yes, `nvshmem_barrier_all_on_stream` with `cudaStreamSynchronize` can be called from the host thread. If multiple barrier synchronization events can happen before synchronizing with the host thread, this gives better performance. Calling `nvshmem_barrier_all` from inside the CUDA kernel can be used for collective synchronization if there are other things that can be done by the same CUDA kernel after a barrier synchronization event. For synchronizing some pairs of PEs and not all, pair-wise `nvshmem_atomic_set` calls by the initiator and `nvshmem_wait_until` or `nvshmem_test` calls by the target can be used.

_Q: How should I allocate memory for NVSHMEM?_

A: See Memory Management for information on allocating symmetric memory. Note that NVSHMEM requires the local and remote pointer to both be symmetric for communication with a remote peer connected by InfiniBand. If the remote peer is P2P accessible (PCI-E or NVLink), the local pointer can be obtained using `cudaMalloc` and is not required to be from the symmetric heap.

_Q: Is there any example of a mini-application written using NVSHMEM?_

A. Yes. The multi-GPU programming models GitHub repository contains an example Jacobi mini-application written using NVSHMEM.

_Q: I am observing degraded performance when running NVSHMEM with NCCL support. How do I fix the performance?_

A: NVSHMEM and NCCL both launch CPU proxy threads for communicatin over IB. These threads can interfere with each other leading to context switches and hence poor performance. These threads should have dedicated hardware threads (or cores) for better performance. NCCL proxy thread is active only when NVSHMEM is using NCCL for collective communication. Try binding options like –bind-to numa or –bind-to core, –bind-to none for improving performance.

_Q: How do I check what environment/configuration variables were used at NVSHMEM build and runtime with application ?_

A: NVSHMEM builds include a binary `nvshmem-info` utility that can be run with `-a` to provide this information. This binary can also be run `-h` option to provide help or usage on how to run this binary.

    ./build/src/bin/nvshmem-info -h
    Print information about NVSHMEM
    Usage: nvshmem-info [options]
    Options:
        -h  This help message
        -a  Print all output
        -n  Print version number
        -b  Print build information
        -e  Print environment variables
        -d  Include hidden environment variables in output
        -r  RST format environment variable output
