# Examples

  * Language Bindings Examples
    * NVSHMEM4Py Examples
      * UID-Based Initialization Example
      * MPI Comm-Based Initialization Example
      * Torch.distributed ProcessGroup Initialization Example
      * Simple P2P Kernel Example
      * On-stream Kernels Example
      * PyTorch and Triton Interoperability Example

Source code for the examples described in this section is available in the examples folder of the NVSHMEM package.

## Attribute-Based Initialization Example

The following code shows an MPI version of the simple shift program that was explained in The NVSHMEM Programming Model. It shows the use of the NVSHMEM attribute-based initialization API where the MPI communicator can be used to set up NVSHMEM.

    #include <stdio.h>
    #include "mpi.h"
    #include "nvshmem.h"
    #include "nvshmemx.h"

    #define CUDA_CHECK(stmt)                                  \
    do {                                                      \
        cudaError_t result = (stmt);                          \
        if (cudaSuccess != result) {                          \
            fprintf(stderr, "[%s:%d] CUDA failed with %s \n", \
             __FILE__, __LINE__, cudaGetErrorString(result)); \
            exit(-1);                                         \
        }                                                     \
    } while (0)

    __global__ void simple_shift(int *destination) {
        int mype = nvshmem_my_pe();
        int npes = nvshmem_n_pes();
        int peer = (mype + 1) % npes;

        nvshmem_int_p(destination, mype, peer);
    }

    int main (int argc, char *argv[]) {
        int mype_node, msg;
        cudaStream_t stream;
        int rank, nranks;
        MPI_Comm mpi_comm = MPI_COMM_WORLD;
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);

        attr.mpi_comm = &mpi_comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
        mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

        CUDA_CHECK(cudaSetDevice(mype_node));
        CUDA_CHECK(cudaStreamCreate(&stream));
        int *destination = (int *) nvshmem_malloc (sizeof(int));

        simple_shift<<<1, 1, 0, stream>>>(destination);
        nvshmemx_barrier_all_on_stream(stream);
        CUDA_CHECK(cudaMemcpyAsync(&msg, destination, sizeof(int),
                    cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("%d: received message %d\n", nvshmem_my_pe(), msg);

        nvshmem_free(destination);
        nvshmem_finalize();
        MPI_Finalize();
        return 0;
    }

The following code shows a Unique ID version of the simple shift program that was explained in The NVSHMEM Programming Model. It shows the use of the NVSHMEM attribute-based initializion API where the Unique ID arguments can be used to set up NVSHMEM.

    #include <stdio.h>
    #include "mpi.h"
    #include "nvshmem.h"
    #include "nvshmemx.h"

    #define CUDA_CHECK(stmt)                                  \
    do {                                                      \
        cudaError_t result = (stmt);                          \
        if (cudaSuccess != result) {                          \
            fprintf(stderr, "[%s:%d] CUDA failed with %s \n", \
             __FILE__, __LINE__, cudaGetErrorString(result)); \
            exit(-1);                                         \
        }                                                     \
    } while (0)

    __global__ void simple_shift(int *destination) {
        int mype = nvshmem_my_pe();
        int npes = nvshmem_n_pes();
        int peer = (mype + 1) % npes;

        nvshmem_int_p(destination, mype, peer);
    }

    int main (int argc, char *argv[]) {
        int mype_node, msg;
        cudaStream_t stream;
        int rank, nranks;
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);

        // PE 0 queries the unique ID
        if (rank == 0) {
           nvshmemx_get_uniqueid(&id);
        }

        // PE 0 broadcast the unique ID to all peers
        MPI_Bcast(&id, sizeof(nvshmemx_uniqueid_t), MPI_UINT8_T, 0, MPI_COMM_WORLD);
        nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
        mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

        CUDA_CHECK(cudaSetDevice(mype_node));
        CUDA_CHECK(cudaStreamCreate(&stream));
        int *destination = (int *) nvshmem_malloc (sizeof(int));

        simple_shift<<<1, 1, 0, stream>>>(destination);
        nvshmemx_barrier_all_on_stream(stream);
        CUDA_CHECK(cudaMemcpyAsync(&msg, destination, sizeof(int),
                    cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        printf("%d: received message %d\n", nvshmem_my_pe(), msg);

        nvshmem_free(destination);
        nvshmem_finalize();
        MPI_Finalize();
        return 0;
    }

## Collective Launch Example

The following code shows an example implementation of a single ring-based reduction where multiple iterations of the code, including computation, communication and synchronization are expressed as a single kernel.

This example also demonstrates the use of NVSHMEM collective launch, required when the NVSHMEM synchronization API is used from inside the CUDA kernel.

There is no MPI dependency for the example. NVSHMEM can be used to port existing MPI applications and develop new applications.

    #include <stdio.h>
    #include "nvshmem.h"
    #include "nvshmemx.h"

    #ifdef NVSHMEM_MPI_SUPPORT
    #include "mpi.h"
    #endif

    #undef CUDA_CHECK
    #define CUDA_CHECK(stmt)                                                          \
        do {                                                                          \
            cudaError_t result = (stmt);                                              \
            if (cudaSuccess != result) {                                              \
                fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                        cudaGetErrorString(result));                                  \
                exit(-1);                                                             \
            }                                                                         \
        } while (0)

    #define NVSHMEM_CHECK(stmt)                                                                \
        do {                                                                                   \
            int result = (stmt);                                                               \
            if (NVSHMEMX_SUCCESS != result) {                                                  \
                fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, \
                        result);                                                               \
                exit(-1);                                                                      \
            }                                                                                  \
        } while (0)

    __global__ void reduce_ring(int *target, int mype, int npes) {
        int peer = (mype + 1) % npes;
        int lvalue = mype;

        for (int i = 1; i < npes; i++) {
            nvshmem_int_p(target, lvalue, peer);
            nvshmem_barrier_all();
            lvalue = *target + mype;
            nvshmem_barrier_all();
        }
    }

    int main(int c, char *v[]) {
        int mype, npes, mype_node;

    #ifdef NVSHMEM_MPI_SUPPORT
        bool use_mpi = false;
        char *value = getenv("NVSHMEMTEST_USE_MPI_LAUNCHER");
        if (value) use_mpi = atoi(value);
    #endif

    #ifdef NVSHMEM_MPI_SUPPORT
        if (use_mpi) {
            MPI_Init(&c, &v);
            int rank, nranks;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &nranks);
            MPI_Comm mpi_comm = MPI_COMM_WORLD;

            nvshmemx_init_attr_t attr;
            attr.mpi_comm = &mpi_comm;
            nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
        } else
            nvshmem_init();
    #else
        nvshmem_init();
    #endif

        mype = nvshmem_my_pe();
        npes = nvshmem_n_pes();
        mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

        // application picks the device each PE will use
        CUDA_CHECK(cudaSetDevice(mype_node));
        int *u = (int *)nvshmem_calloc(1, sizeof(int));
        int *h = (int *)calloc(1, sizeof(int));

        void *args[] = {&u, &mype, &npes};
        dim3 dimBlock(1);
        dim3 dimGrid(1);

        NVSHMEM_CHECK(
            nvshmemx_collective_launch((const void *)reduce_ring, dimGrid, dimBlock, args, 0, 0));
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaMemcpy(h, u,  sizeof(int), cudaMemcpyDeviceToHost);
        printf("results on device [%d] is %d \n",mype, h[0]);

        nvshmem_free(u);
        free(h);
        nvshmem_finalize();

    #ifdef NVSHMEM_MPI_SUPPORT
        if (use_mpi) MPI_Finalize();
    #endif

        return 0;
    }

## On-Stream Example

The following example shows how `nvshmemx_*_on_stream` functions can be used to enqueue a SHMEM operation onto a CUDA stream for execution in stream order. Specifically, the example shows the following:

>   * How a collective SHMEM reduction operation can be made to wait on a preceding kernel in the stream.
>   * How a kernel can be made to wait for a communication result from a previous collective SHMEM reduction operation.
>

The example shows one use case for relieving CPU control over GPU compute and communication.

    #include <stdio.h>
    #include "nvshmem.h"
    #include "nvshmemx.h"

    #ifdef NVSHMEM_MPI_SUPPORT
    #include "mpi.h"
    #endif

    #define THRESHOLD 42
    #define CORRECTION 7

    #undef CUDA_CHECK
    #define CUDA_CHECK(stmt)                                                          \
        do {                                                                          \
            cudaError_t result = (stmt);                                              \
            if (cudaSuccess != result) {                                              \
                fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                        cudaGetErrorString(result));                                  \
                exit(-1);                                                             \
            }                                                                         \
        } while (0)

    __global__ void accumulate(int *input, int *partial_sum) {
        int index = threadIdx.x;
        if (0 == index) *partial_sum = 0;
        __syncthreads();
        atomicAdd(partial_sum, input[index]);
    }

    __global__ void correct_accumulate(int *input, int *partial_sum, int *full_sum) {
        int index = threadIdx.x;
        if (*full_sum > THRESHOLD) {
            input[index] = input[index] - CORRECTION;
        }
        if (0 == index) *partial_sum = 0;
        __syncthreads();
        atomicAdd(partial_sum, input[index]);
    }

    int main(int c, char *v[]) {
        int mype, npes, mype_node;
        int *input;
        int *partial_sum;
        int *full_sum;
        int input_nelems = 512;
        int to_all_nelems = 1;
        cudaStream_t stream;

    #ifdef NVSHMEM_MPI_SUPPORT
        bool use_mpi = false;
        char *value = getenv("NVSHMEMTEST_USE_MPI_LAUNCHER");
        if (value) use_mpi = atoi(value);
    #endif

    #ifdef NVSHMEM_MPI_SUPPORT
        if (use_mpi) {
            MPI_Init(&c, &v);
            int rank, nranks;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &nranks);
            MPI_Comm mpi_comm = MPI_COMM_WORLD;

            nvshmemx_init_attr_t attr;
            attr.mpi_comm = &mpi_comm;
            nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
        } else
            nvshmem_init();
    #else
        nvshmem_init();
    #endif

        mype = nvshmem_my_pe();
        npes = nvshmem_n_pes();
        mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
        CUDA_CHECK(cudaSetDevice(mype_node));
        CUDA_CHECK(cudaStreamCreate(&stream));

        input = (int *)nvshmem_malloc(sizeof(int) * input_nelems);
        partial_sum = (int *)nvshmem_malloc(sizeof(int));
        full_sum = (int *)nvshmem_malloc(sizeof(int));

        accumulate<<<1, input_nelems, 0, stream>>>(input, partial_sum);
        nvshmemx_int_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, full_sum, partial_sum, to_all_nelems, stream);
        correct_accumulate<<<1, input_nelems, 0, stream>>>(input, partial_sum, full_sum);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        printf("[%d of %d] run complete \n", mype, npes);

        CUDA_CHECK(cudaStreamDestroy(stream));

        nvshmem_free(input);
        nvshmem_free(partial_sum);
        nvshmem_free(full_sum);

        nvshmem_finalize();

    #ifdef NVSHMEM_MPI_SUPPORT
        if (use_mpi) MPI_Finalize();
    #endif
        return 0;
    }

## Threadgroup Example

The example in this section shows how `nvshmemx_collect32_block` can be used to leverage threads to accelerate a SHMEM collect operation when all threads in the block depend on the result of a preceding communication operation. For this instance, partial vector sums are computed across different PEs and have a SHMEM collect operation to obtain the complete sum across PEs.

    #include <stdio.h>
    #include "nvshmem.h"
    #include "nvshmemx.h"

    #ifdef NVSHMEM_MPI_SUPPORT
    #include "mpi.h"
    #endif

    #define NTHREADS 512

    #undef CUDA_CHECK
    #define CUDA_CHECK(stmt)                                                          \
        do {                                                                          \
            cudaError_t result = (stmt);                                              \
            if (cudaSuccess != result) {                                              \
                fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                        cudaGetErrorString(result));                                  \
                exit(-1);                                                             \
            }                                                                         \
        } while (0)

    __global__ void distributed_vector_sum(int *x, int *y, int *partial_sum, int *sum,
                                           int use_threadgroup, int mype, int npes) {
        int index = threadIdx.x;
        int nelems = blockDim.x;
        partial_sum[index] = x[index] + y[index];

        if (use_threadgroup) {
            /* all threads realize the entire fcollect operation */
            nvshmemx_int_fcollect_block(NVSHMEM_TEAM_WORLD, sum, partial_sum, nelems);
        } else {
            /* thread 0 realizes the entire fcollect operation */
            if (0 == index) {
                nvshmem_int_fcollect(NVSHMEM_TEAM_WORLD, sum, partial_sum, nelems);
            }
        }
    }

    int main(int c, char *v[]) {
        int mype, npes, mype_node;
        int *x;
        int *y;
        int *partial_sum;
        int *sum;
        int use_threadgroup = 1;
        int nthreads = NTHREADS;

    #ifdef NVSHMEM_MPI_SUPPORT
        bool use_mpi = false;
        char *value = getenv("NVSHMEMTEST_USE_MPI_LAUNCHER");
        if (value) use_mpi = atoi(value);
    #endif

    #ifdef NVSHMEM_MPI_SUPPORT
        if (use_mpi) {
            MPI_Init(&c, &v);
            int rank, nranks;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &nranks);
            MPI_Comm mpi_comm = MPI_COMM_WORLD;

            nvshmemx_init_attr_t attr;
            attr.mpi_comm = &mpi_comm;
            nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
        } else
            nvshmem_init();
    #else
        nvshmem_init();
    #endif

        npes = nvshmem_n_pes();
        mype = nvshmem_my_pe();
        mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

        CUDA_CHECK(cudaSetDevice(mype_node));

        x = (int *)nvshmem_malloc(sizeof(int) * nthreads);
        y = (int *)nvshmem_malloc(sizeof(int) * nthreads);
        partial_sum = (int *)nvshmem_malloc(sizeof(int) * nthreads);
        sum = (int *)nvshmem_malloc(sizeof(int) * nthreads * npes);

        void *args[] = {&x, &y, &partial_sum, &sum, &use_threadgroup, &mype, &npes};
        dim3 dimBlock(nthreads);
        dim3 dimGrid(1);
        nvshmemx_collective_launch((const void *)distributed_vector_sum, dimGrid, dimBlock, args, 0, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("[%d of %d] run complete \n", mype, npes);

        nvshmem_free(x);
        nvshmem_free(y);
        nvshmem_free(partial_sum);
        nvshmem_free(sum);

        nvshmem_finalize();
    #ifdef NVSHMEM_MPI_SUPPORT
        if (use_mpi) MPI_Finalize();
    #endif

        return 0;
    }

## Put on Block Example

In the example below, every thread in block 0 calls `nvshmemx_float_put_block`. Alternatively, every thread can call `nvshmem_float_p`, but `nvshmem_float_p` has a disadvantage that when the destination GPU is connected via InfiniBand, there is one RMA message for every single element, which can be detrimental to performance.

The disadvantage with using `nvshmem_float_put` in this case is that when the destination GPU is P2P-connected, a single thread will copy the entire data to the destination GPU. While `nvshmemx_float_put_block` can leverage all the threads in the block to copy the data in parallel to the destination GPU.

    #include <stdio.h>
    #include <assert.h>
    #include "nvshmem.h"
    #include "nvshmemx.h"

    #undef CUDA_CHECK
    #define CUDA_CHECK(stmt)                                                          \
        do {                                                                          \
            cudaError_t result = (stmt);                                              \
            if (cudaSuccess != result) {                                              \
                fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                        cudaGetErrorString(result));                                  \
                exit(-1);                                                             \
            }                                                                         \
        } while (0)

    #define THREADS_PER_BLOCK 1024

    __global__ void set_and_shift_kernel(float *send_data, float *recv_data, int num_elems, int mype,
                                         int npes) {
        int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
        /* set the corresponding element of send_data */
        if (thread_idx < num_elems) send_data[thread_idx] = mype;

        int peer = (mype + 1) % npes;
        /* Every thread in block 0 calls nvshmemx_float_put_block. Alternatively,
           every thread can call shmem_float_p, but shmem_float_p has a disadvantage
           that when the destination GPU is connected via IB, there will be one rma
           message for every single element which can be detrimental to performance.
           And the disadvantage with shmem_float_put is that when the destination GPU is p2p
           connected, it cannot leverage multiple threads to copy the data to the destination
           GPU. */
        int block_offset = blockIdx.x * blockDim.x;
        nvshmemx_float_put_block(recv_data + block_offset, send_data + block_offset,
                                 min(blockDim.x, num_elems - block_offset),
                                 peer); /* All threads in a block call the API
                                           with the same arguments */
    }

    int main(int c, char *v[]) {
        int mype, npes, mype_node;
        float *send_data, *recv_data;
        int num_elems = 8192;
        int num_blocks;
        cudaStream_t stream;

        nvshmem_init();

        mype = nvshmem_my_pe();
        npes = nvshmem_n_pes();
        mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

        // application picks the device each PE will use
        CUDA_CHECK(cudaSetDevice(mype_node));
        CUDA_CHECK(cudaStreamCreate(&stream));
        send_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);
        recv_data = (float *)nvshmem_malloc(sizeof(float) * num_elems);
        assert(send_data != NULL && recv_data != NULL);

        assert(num_elems % THREADS_PER_BLOCK == 0); /* for simplicity */
        num_blocks = num_elems / THREADS_PER_BLOCK;

        set_and_shift_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(send_data, recv_data, num_elems, mype, npes);
        nvshmemx_barrier_all_on_stream(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        /* Do data validation */
        float *host = new float[num_elems];
        CUDA_CHECK(cudaMemcpy(host, recv_data, num_elems * sizeof(float), cudaMemcpyDefault));
        int ref = (mype - 1 + npes) % npes;
        bool success = true;
        for (int i = 0; i < num_elems; ++i) {
            if (host[i] != ref) {
                printf("Error at %d of rank %d: %f\n", i, mype, host[i]);
                success = false;
                break;
            }
        }

        if (success) {
            printf("[%d of %d] run complete \n", mype, npes);
        } else {
            printf("[%d of %d] run failure \n", mype, npes);
        }

        nvshmem_free(send_data);
        nvshmem_free(recv_data);

        nvshmem_finalize();

        return 0;
    }

## TMA Put Example

The example below registers CTA shared memory so NVSHMEM can use TMA for eligible block-scoped put operations over NVLink. Run with `NVSHMEM_TMA_POLICY=ENABLE` on supported systems. For details, see Using TMA with NVSHMEM.

    #include <stdio.h>
    #include <stdlib.h>
    #include <cuda_runtime.h>
    #include "nvshmem.h"
    #include "nvshmemx.h"

    #undef CUDA_CHECK
    #define CUDA_CHECK(stmt)                                                          \
        do {                                                                          \
            cudaError_t result = (stmt);                                              \
            if (cudaSuccess != result) {                                              \
                fprintf(stderr, "[%s:%d] cuda failed with %s\n", __FILE__, __LINE__,  \
                        cudaGetErrorString(result));                                  \
                nvshmem_global_exit(1);                                               \
            }                                                                         \
        } while (0)

    #define NUM_ELEMS 1024
    #define THREADS_PER_BLOCK 128

    __global__ void tma_put_kernel(float *send_data, float *recv_data, int num_elems,
                                   int mype, int npes, size_t smem_size) {
        extern __shared__ char nvshmem_smem[];

        if (threadIdx.x == 0) {
            nvshmemx_give_smem(nvshmem_smem, smem_size);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < num_elems; i += blockDim.x) {
            send_data[i] = (float)(mype * 1000 + i);
        }
        __syncthreads();

        int peer = (mype + 1) % npes;
        nvshmemx_float_put_nbi_block(recv_data, send_data, num_elems, peer);
        nvshmem_quiet();

        __syncthreads();
        if (threadIdx.x == 0) {
            nvshmemx_release_smem();
        }
    }

    int main(void) {
        int mype, npes, mype_node;
        float *send_data, *recv_data;
        cudaStream_t stream;

        nvshmem_init();

        mype = nvshmem_my_pe();
        npes = nvshmem_n_pes();
        mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

        CUDA_CHECK(cudaSetDevice(mype_node));
        CUDA_CHECK(cudaStreamCreate(&stream));

        send_data = (float *)nvshmem_malloc(sizeof(float) * NUM_ELEMS);
        recv_data = (float *)nvshmem_malloc(sizeof(float) * NUM_ELEMS);
        if (send_data == NULL || recv_data == NULL) {
            fprintf(stderr, "[%d] nvshmem_malloc failed\n", mype);
            nvshmem_global_exit(1);
            return 1;
        }

        CUDA_CHECK(cudaMemset(recv_data, 0, sizeof(float) * NUM_ELEMS));
        nvshmem_barrier_all();

        size_t smem_size = (size_t)nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
        if (smem_size > 48 * 1024) {
            CUDA_CHECK(cudaFuncSetAttribute(tma_put_kernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            (int)smem_size));
        }

        tma_put_kernel<<<1, THREADS_PER_BLOCK, smem_size, stream>>>(
            send_data, recv_data, NUM_ELEMS, mype, npes, smem_size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        nvshmem_barrier_all();

        float host[NUM_ELEMS];
        CUDA_CHECK(cudaMemcpy(host, recv_data, sizeof(host), cudaMemcpyDefault));

        int expected_pe = (mype - 1 + npes) % npes;
        bool success = true;
        for (int i = 0; i < NUM_ELEMS; i++) {
            float expected = (float)(expected_pe * 1000 + i);
            if (host[i] != expected) {
                printf("[%d] ERROR at %d: got %f, expected %f\n",
                       mype, i, host[i], expected);
                success = false;
                break;
            }
        }

        if (success) {
            printf("[%d of %d] TMA put example passed\n", mype, npes);
        }

        nvshmem_free(send_data);
        nvshmem_free(recv_data);
        CUDA_CHECK(cudaStreamDestroy(stream));
        nvshmem_finalize();

        return success ? 0 : 1;
    }

## Threadgroup Fence and Signal Example

In the example below, every thread in a CTA participates in a block-scoped nonblocking put. After the CTA synchronizes, one thread calls `nvshmem_fence` and then signals the peer. The fence orders the signal after the put operation issued by the CTA; not every issuing thread needs to call `nvshmem_fence`. If the program needs completion instead of ordering before a signal, the same single-thread-after-synchronization pattern can be used with `nvshmem_quiet`.

    #include <stdio.h>
    #include <stdlib.h>
    #include <stdint.h>
    #include <cuda_runtime.h>
    #include "nvshmem.h"
    #include "nvshmemx.h"

    #undef CUDA_CHECK
    #define CUDA_CHECK(stmt)                                                          \
        do {                                                                          \
            cudaError_t result = (stmt);                                              \
            if (cudaSuccess != result) {                                              \
                fprintf(stderr, "[%s:%d] cuda failed with %s\n", __FILE__, __LINE__,  \
                        cudaGetErrorString(result));                                  \
                nvshmem_global_exit(1);                                               \
            }                                                                         \
        } while (0)

    #undef NVSHMEM_CHECK
    #define NVSHMEM_CHECK(stmt)                                                       \
        do {                                                                          \
            int result = (stmt);                                                      \
            if (result != 0) {                                                        \
                fprintf(stderr, "[%s:%d] nvshmem failed with error %d\n", __FILE__,   \
                        __LINE__, result);                                            \
                nvshmem_global_exit(1);                                               \
            }                                                                         \
        } while (0)

    #define NUM_ELEMS 1024
    #define THREADS_PER_BLOCK 256

    __global__ void put_fence_signal_kernel(int *send_data, int *recv_data,
                                            uint64_t *signal, int num_elems,
                                            int mype, int npes) {
        int peer = (mype + 1) % npes;
        int tid = threadIdx.x;

        for (int i = tid; i < num_elems; i += blockDim.x) {
            send_data[i] = mype * 1000 + i;
        }
        __syncthreads();

        nvshmemx_int_put_nbi_block(recv_data, send_data, num_elems, peer);

        /*
         * The block sync ensures all threads have issued the block-scoped put
         * before thread 0 performs the ordering operation and signal.
         */
        __syncthreads();

        if (tid == 0) {
            nvshmem_fence();
            nvshmemx_signal_op(signal, 1, NVSHMEM_SIGNAL_SET, peer);
            nvshmem_signal_wait_until(signal, NVSHMEM_CMP_EQ, 1);
        }

        __syncthreads();
    }

    int main(void) {
        int mype, npes, mype_node;
        int num_elems = NUM_ELEMS;
        int *send_data, *recv_data;
        uint64_t *signal;

        nvshmem_init();

        mype = nvshmem_my_pe();
        npes = nvshmem_n_pes();
        mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

        CUDA_CHECK(cudaSetDevice(mype_node));

        send_data = (int *)nvshmem_malloc(sizeof(int) * NUM_ELEMS);
        recv_data = (int *)nvshmem_malloc(sizeof(int) * NUM_ELEMS);
        signal = (uint64_t *)nvshmem_calloc(1, sizeof(uint64_t));
        if (send_data == NULL || recv_data == NULL || signal == NULL) {
            fprintf(stderr, "[%d] nvshmem allocation failed\n", mype);
            nvshmem_global_exit(1);
        }

        CUDA_CHECK(cudaMemset(recv_data, 0, sizeof(int) * NUM_ELEMS));
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        void *args[] = {&send_data, &recv_data, &signal, &num_elems, &mype, &npes};
        dim3 grid(1);
        dim3 block(THREADS_PER_BLOCK);

        NVSHMEM_CHECK(nvshmemx_collective_launch((const void *)put_fence_signal_kernel,
                                                 grid, block, args, 0, 0));
        CUDA_CHECK(cudaDeviceSynchronize());

        int host[NUM_ELEMS];
        CUDA_CHECK(cudaMemcpy(host, recv_data, sizeof(host), cudaMemcpyDefault));

        int expected_pe = (mype - 1 + npes) % npes;
        bool success = true;
        for (int i = 0; i < NUM_ELEMS; i++) {
            int expected = expected_pe * 1000 + i;
            if (host[i] != expected) {
                printf("[%d] ERROR at %d: got %d, expected %d\n",
                       mype, i, host[i], expected);
                success = false;
                break;
            }
        }

        if (success) {
            printf("[%d of %d] fence and signal example passed\n", mype, npes);
        }

        nvshmem_free(send_data);
        nvshmem_free(recv_data);
        nvshmem_free(signal);

        nvshmem_finalize();
        return success ? 0 : 1;
    }

## Ring Broadcast Example

In the example below, PE 0 broadcasts a message by sending it to PE 1, which sends the message to PE 2 and so on. This example demonstrates several NVSHMEM APIs, including the use of `nvshmem_fence` to order communication and `nvshmem_signal_wait_until` and `nvshmemx_signal_op` for point-to-point synchronization.

    #include <stdio.h>
    #include <stdint.h>
    #include <cuda.h>
    #include <nvshmem.h>
    #include <nvshmemx.h>

    __global__ void ring_bcast(int *data, size_t nelem, int root, uint64_t *psync) {
        int mype = nvshmem_my_pe();
        int npes = nvshmem_n_pes();
        int peer = (mype + 1) % npes;

        if (mype == root)
            *psync = 1;

        nvshmem_signal_wait_until(psync, NVSHMEM_CMP_NE, 0);

        if (mype == npes-1) return;

        nvshmem_int_put(data, data, nelem, peer);
        nvshmem_fence();
        nvshmemx_signal_op(psync, 1, NVSHMEM_SIGNAL_SET, peer);

        *psync = 0;
    }

    int main(void) {
        size_t data_len = 32;
        cudaStream_t stream;

        nvshmem_init();

        int mype      = nvshmem_my_pe();
        int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

        cudaSetDevice(mype_node);
        cudaStreamCreate(&stream);

        int      *data   = (int *) nvshmem_malloc(sizeof(int) * data_len);
        int      *data_h = (int *) malloc(sizeof(int) * data_len);
        uint64_t *psync  = (uint64_t *) nvshmem_calloc(1, sizeof(uint64_t));

        for (size_t i = 0; i < data_len; i++)
            data_h[i] = mype + i;

        cudaMemcpyAsync(data, data_h, sizeof(int) * data_len, cudaMemcpyHostToDevice, stream);

        int   root = 0;
        dim3  gridDim(1), blockDim(1);
        void *args[] = { &data, &data_len, &root, &psync };

        nvshmemx_barrier_all_on_stream(stream);
        nvshmemx_collective_launch((const void *)ring_bcast, gridDim, blockDim, args, 0, stream);
        nvshmemx_barrier_all_on_stream(stream);
        cudaMemcpyAsync(data_h, data, sizeof(int) * data_len, cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        for (size_t i = 0; i < data_len; i++) {
            if (data_h[i] != i)
                printf("PE %d error, data[%zu] = %d expected data[%zu] = %d\n",
                        mype, i, data_h[i], i, (int) i);
        }

        nvshmem_free(data);
        nvshmem_free(psync);
        free(data_h);

        nvshmem_finalize();
        return 0;
    }

## Ring Allreduce Example

In the example below, PE0 receives a message chunk from its left neighbor, performs a local reduction and sends the resulting chunk to its right neighbor (PE1), and so on. Eventually, every PE (but last) broadcast its own chunk to right neighbor. This examples demonstrates several NVSHMEM APIs, including the use of `nvshmem_int_put_signal_nbi` and `nvshmem_signal_wait_until` for point-to-point communication & synchronization.

    /*
     * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
     *
     * NVIDIA CORPORATION and its licensors retain all intellectual property
     * and proprietary rights in and to this software, related documentation
     * and any modifications thereto.  Any use, reproduction, disclosure or
     * distribution of this software and related documentation without an express
     * license agreement from NVIDIA CORPORATION is strictly prohibited.
     *
     * See COPYRIGHT.txt for license information
     */

    /* This example performs an allreduce operation using ring algorithm when
       GPUs are connected via remote interconect like IB/RoCE/EFA, etc.
       It does ring reduce followed by ring broadcast. We use single threaded put_signal API
       as single thread is sufficient for remote transfers. The example is expected
       to be performant only when GPUs are connected via remote interconnect. */

    #include <stdio.h>
    #include <stdint.h>
    #include <cuda.h>
    #include <nvshmem.h>
    #include <nvshmemx.h>
    #include <unistd.h>
    #include <ctype.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>

    #undef CUDA_CHECK
    #define CUDA_CHECK(stmt)                                                          \
        do {                                                                          \
            cudaError_t result = (stmt);                                              \
            if (cudaSuccess != result) {                                              \
                fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                        cudaGetErrorString(result));                                  \
                exit(-1);                                                             \
            }                                                                         \
        } while (0)

    /* atol() + optional scaled suffix recognition: 1K, 2M, 3G, 1T */
    static inline int atol_scaled(const char *str, size_t *out) {
        int scale, n;
        double p = -1.0;
        char f;
        n = sscanf(str, "%lf%c", &p, &f);

        if (n == 2) {
            switch (f) {
                case 'k':
                case 'K':
                    scale = 10;
                    break;
                case 'm':
                case 'M':
                    scale = 20;
                    break;
                case 'g':
                case 'G':
                    scale = 30;
                    break;
                case 't':
                case 'T':
                    scale = 40;
                    break;
                default:
                    return 1;
            }
        } else if (p < 0) {
            return 1;
        } else
            scale = 0;

        *out = (size_t)ceil(p * (1lu << scale));
        return 0;
    }

    size_t min_size = 1024 * 1024 * 32;
    size_t max_size = min_size * 16;
    size_t num_blocks = 32;
    size_t threads_per_block = 512;
    size_t iters = 4;
    size_t warmup_iters = 1;
    size_t step_factor = 2;
    size_t chunk_size = 262144;

    // perform Allreduce using ring
    __global__ void ring_reduce(int *dst, const int *src, size_t nreduce, uint64_t *signal,
                                size_t chunk_size) {
        int mype = nvshmem_my_pe();
        int npes = nvshmem_n_pes();
        int peer = (mype + 1) % npes;

        int thread_id = threadIdx.x;
        int num_threads = blockDim.x;
        int num_blocks = gridDim.x;
        int block_idx = blockIdx.x;
        size_t elems_per_block = nreduce / num_blocks;

        // Change src, dst, nreduce, signal to what this block is going to process
        // Each CTA will work independently
        if (elems_per_block * (blockIdx.x + 1) > nreduce) return;
        src = src + block_idx * elems_per_block;
        dst = dst + block_idx * elems_per_block;
        nreduce = elems_per_block;
        signal = signal + block_idx;

        size_t chunk_elems = chunk_size / sizeof(int);
        size_t num_chunks = nreduce / chunk_elems;

        // reduce phase
        for (size_t chunk = 0; chunk < num_chunks; chunk++) {
            if (mype != 0) {
                if (thread_id == 0) nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE, chunk + 1);

                __syncthreads();
                for (size_t i = thread_id; i < chunk_elems; i += num_threads) {
                    dst[i] = dst[i] + src[i];
                }
                __syncthreads();
            }
            if (thread_id == 0)
                nvshmem_int_put_signal_nbi(dst, (mype == 0) ? src : dst, chunk_elems, signal, 1,
                                           NVSHMEM_SIGNAL_ADD, peer);
            src = src + chunk_elems;
            dst = dst + chunk_elems;
        }

        // Broadcast phase
        dst = dst - num_chunks * chunk_elems;
        if (thread_id == 0) {
            for (size_t chunk = 0; chunk < num_chunks; chunk++) {
                if (mype < npes - 1) {  // Last pe already has the final result
                    nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE,
                                              (mype == 0) ? chunk + 1 : num_chunks + chunk + 1);
                }
                if (mype < npes - 2)
                    nvshmem_int_put_signal_nbi(dst, dst, chunk_elems, signal, 1, NVSHMEM_SIGNAL_ADD,
                                               peer);
                dst = dst + chunk_elems;
            }
            *signal = 0;  // reset for next iteration
        }
    }

    int main(int argc, char **argv) {
        int c;
        while ((c = getopt(argc, argv, "b:e:f:n:w:c:t:m:")) != -1) {
            switch (c) {
                case 'b':
                    atol_scaled(optarg, &min_size);
                    break;
                case 'e':
                    atol_scaled(optarg, &max_size);
                    break;
                case 'f':
                    atol_scaled(optarg, &step_factor);
                    break;
                case 'n':
                    atol_scaled(optarg, &iters);
                    break;
                case 'w':
                    atol_scaled(optarg, &warmup_iters);
                    break;
                case 'c':
                    atol_scaled(optarg, &num_blocks);
                    break;
                case 't':
                    atol_scaled(optarg, &threads_per_block);
                    break;
                case 'm':
                    atol_scaled(optarg, &chunk_size);
                    break;
                case '?':
                    if (optopt == 'c')
                        fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                    else if (isprint(optopt))
                        fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                    else
                        fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
                    return 1;
                default:
                    abort();
            }
        }
        size_t min_ints = min_size / sizeof(int);
        assert(min_ints % num_blocks == 0);

        nvshmem_init();

        int mype = nvshmem_my_pe();
        int npes = nvshmem_n_pes();
        int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
        cudaStream_t stream;
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaSetDevice(mype_node));
        CUDA_CHECK(cudaStreamCreate(&stream));

        size_t max_ints = max_size / sizeof(int);
        int *dst = (int *)nvshmem_malloc(max_size);
        int *src = (int *)nvshmem_malloc(max_size);
        int *data_h = (int *)malloc(max_size);
        uint64_t *signal = (uint64_t *)nvshmem_calloc(num_blocks, sizeof(uint64_t));
        dim3 gridDim(num_blocks), blockDim(threads_per_block);

        for (size_t i = 0; i < max_ints; i++) data_h[i] = i;

        CUDA_CHECK(cudaMemcpyAsync(src, data_h, max_size, cudaMemcpyHostToDevice, stream));
        nvshmemx_barrier_all_on_stream(stream);

        for (size_t size = min_size; size <= max_size; size *= step_factor) {
            size_t num_ints = size / sizeof(int);
            void *args[] = {&dst, &src, &num_ints, &signal, &chunk_size};

            // do warmup
            for (size_t i = 0; i < warmup_iters; i++) {
                nvshmemx_collective_launch((const void *)ring_reduce, gridDim, blockDim, args, 0,
                                           stream);
                nvshmemx_barrier_all_on_stream(stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // main loop
            CUDA_CHECK(cudaEventRecord(start, stream));
            for (size_t i = 0; i < iters; i++) {
                nvshmemx_collective_launch((const void *)ring_reduce, gridDim, blockDim, args, 0,
                                           stream);
                nvshmemx_barrier_all_on_stream(stream);
            }
            CUDA_CHECK(cudaEventRecord(stop, stream));

            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (!mype) {
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                printf("%zuB \t %fms\n", size, ms / iters);
            }

            // validate output
            CUDA_CHECK(cudaMemcpy(data_h, dst, size, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < num_ints; i++) {
                if (data_h[i] != (int)i * npes)
                    printf("PE %d error, data[%zu] = %d expected data[%zu] = %d\n", mype, i, data_h[i],
                           i, (int)i * npes);
            }
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        nvshmem_free(dst);
        nvshmem_free(src);
        nvshmem_free(signal);
        free(data_h);

        nvshmem_finalize();
        return 0;
    }

## User Buffer Registration Example

The example below shows user allocated memory buffer being registered with NVSHMEM symmetric heap and used for communication. The example uses CUDA VMM APIs to allocate memory buffer (`createUserBuffer` function) and registers it to the heap using `nvshmemx_buffer_register_symmetric` routine. Once registered, the buffer can be used for communication like any memory allocated using `nvshmem_malloc` and/or its friends.

    #include <stdio.h>
    #include <iostream>
    #include <stdlib.h>
    #include <unistd.h>
    #include "nvshmem.h"
    #include "nvshmemx.h"
    #include "cuda_runtime.h"

    #define GRANULARITY 536870912UL
    #define COLL_NELEMS 4096

    #undef CUDA_CHECK
    #define CUDA_CHECK(stmt)                                                          \
        do {                                                                          \
            cudaError_t result = (stmt);                                              \
            if (cudaSuccess != result) {                                              \
                fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                        cudaGetErrorString(result));                                  \
                exit(-1);                                                             \
            }                                                                         \
            assert(cudaSuccess == result);                                            \
        } while (0)

    #undef CU_CHECK
    #define CU_CHECK(stmt)                                                                  \
        do {                                                                                \
            CUresult result = (stmt);                                                       \
            const char *str;                                                                \
            if (CUDA_SUCCESS != result) {                                                   \
                CUresult ret = cuGetErrorString(result, &str);                              \
                if (ret == CUDA_ERROR_INVALID_VALUE) str = "Unknown error";                 \
                fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, str); \
                exit(-1);                                                                   \
            }                                                                               \
            assert(CUDA_SUCCESS == result);                                                 \
        } while (0)

    __global__ void init_data_kernel(float *source, size_t nelems) {
        for (int i = 0; i < nelems; ++i) {
            source[i] = (float)i;
        }
    }

    void *createUserBuffer(size_t size, CUmemAllocationProp &prop) {
        void *bufAddr = nullptr;

        CUmemAccessDesc accessDescriptor;
        accessDescriptor.location.id = prop.location.id;
        accessDescriptor.location.type = prop.location.type;
        accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        CUmemGenericAllocationHandle userAllocHandle;

        CU_CHECK(cuMemCreate(&userAllocHandle, size, (const CUmemAllocationProp *)&prop, 0));
        CU_CHECK(cuMemAddressReserve((CUdeviceptr *)&bufAddr, size, 0, (CUdeviceptr)NULL, 0));
        CU_CHECK(cuMemMap((CUdeviceptr)bufAddr, size, 0, userAllocHandle, 0));
        CU_CHECK(
            cuMemSetAccess((CUdeviceptr)bufAddr, size, (const CUmemAccessDesc *)&accessDescriptor, 1));
        return bufAddr;
    }

    void releaseUserBuf(void *ptr, size_t size) {
        CUmemGenericAllocationHandle memHandle;
        CU_CHECK(cuMemRetainAllocationHandle(&memHandle, ptr));
        CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
        CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
        CU_CHECK(cuMemRelease(memHandle));
    }

    int main(int argc, char **argv) {
        nvshmem_init();
        int status = 0;
        int mype, npes;
        int npes_node, mype_node;
        const size_t size = GRANULARITY;
        void *buffer;
        void *mmaped_buffer;
        CUmemAllocationProp prop = {};
        int dev_id;
        float *source, *dest, *dest_h;
        size_t nelems;
        cudaStream_t stream;
        nvshmem_team_t team = NVSHMEM_TEAM_WORLD;

        mype = nvshmem_my_pe();
        mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
        npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
        npes = nvshmem_n_pes();
        dev_id = mype_node % npes_node;
        CUDA_CHECK(cudaSetDevice(dev_id));

        if (!mype) printf("creating and mmapping buffer of size: %lu\n", size);
        // Allocation of user buffer is local
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = dev_id;
        prop.allocFlags.gpuDirectRDMACapable = 1;
        prop.requestedHandleTypes =
            (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);

        buffer = createUserBuffer(size, prop);
        if (!buffer) {
            fprintf(stderr, "Failed to create user buffer \n");
            status = 1;
            goto out;
        }
        mmaped_buffer = (void *)nvshmemx_buffer_register_symmetric(buffer, size, 0);
        if (!mmaped_buffer) {
            fprintf(stderr, "shmem_mmap failed \n");
            status = 1;
            goto out;
        }
        CUDA_CHECK(cudaMemset(mmaped_buffer, 0, size));

        // test heap usage to verify mmap correctness
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        nelems = COLL_NELEMS;
        nelems = nelems / 2;  // split the buffer into source and dest

        source = (float *)mmaped_buffer;
        dest = (float *)(mmaped_buffer) + nelems;
        dest_h = (float *)malloc(nelems * sizeof(float));

        init_data_kernel<<<1, 1, 0, stream>>>((float *)source, nelems);
        nvshmemx_barrier_on_stream(team, stream);
        nvshmemx_float_sum_reduce_on_stream(team, (float *)dest, (const float *)source, nelems, stream);
        cudaStreamSynchronize(stream);

        CUDA_CHECK(cudaMemcpy(dest_h, dest, nelems * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < nelems; i++) {
            if (dest_h[i] != (float)i * npes) {
                printf("PE %d error, data[%zu] = %f expected data[%zu] = %f\n", mype, i, dest_h[i], i,
                       (float)i * npes);
                status = -1;
            }
        }
        if (!status) {
            fprintf(stderr, "No errors found\n");
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        // free all buffers
        status = nvshmemx_buffer_unregister_symmetric(mmaped_buffer, size);
        if (status) {
            fprintf(stderr, "nvshmemx_buffer_unregister_symmetric failed\n");
        }
        free(dest_h);
        nvshmem_finalize();
        releaseUserBuf(buffer, size);
    out:
        return status;
    }

## GEMM + AllReduce Fused Kernel Example

This example uses the Tile-granular APIs to create a fused GEMM + AllReduce kernel. This example builds on Blackwell fp16 GEMM from CUTLASS and integrates AllReduce within the kernel. Below we show a code snippet from the host code which sets up the input matrices and launches the fused kernel. The complete example can be found at /examples/gemm_allreduce. As this example builds on top of CUTLASS example, it is recommended to go over the CUTLASS example. Specifically, this example extends the cutlass/examples/70_blackwell_gemm/70_blackwell_fp16_gemm.cu to perform tensor allocation in NVSHMEM symmetric heap and perform AllReduce collective at tile-granularity (please refer `gemmAR_fusion_blackwell_fp16.cu`). The TMA warp specialized GEMM kernel from CUTLASS is modified to invoke AllReduce collective operation on each output tile (please refer to `sm100_gemm_tma_warpspecialized_allreduce.hpp`).

For integrating AllReduce, a new class `CollectiveAllReduceMulticastWarpSpecialized` (please refer to allreduce_nvls_warpspecilaized.hpp) is added which performs the tile collective operations and contains the required inputs and helper functions. `do_allreduce()` function is called after every tile of GEMM output is computed, this function takes in the coordinates of the computed tile along with matrix data pointers to create `nvshmemx::Tensors` on which AllReduce is performed. `tile_collective_wait()` ensures completion of the collectives.

A new allocator class `nvshmemAllocation` is also added (please refer `nvshmemAlloc.hpp`) which allows allocating tensors from within NVSHMEM symmetric heap.

    /////////////////////////////////////////////////////////////////////
    /// GEMM kernel configurations
    /////////////////////////////////////////////////////////////////////

    // A matrix configuration
    using ElementA = half_t;         // Element type for A matrix operand
    using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

    // B matrix configuration
    using ElementB = half_t;
    using LayoutB = cutlass::layout::ColumnMajor;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

    // C/D matrix configuration
    using ElementC = float;
    using LayoutC = cutlass::layout::ColumnMajor;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    // Kernel functional config
    using ElementAccumulator = float;
    using ArchTag = cutlass::arch::Sm100;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    using MmaTileShape_MNK = Shape<_256, _128, _64>;
    // Shape of the threadblocks in a cluster
    using ClusterShape_MNK = Shape<_2, _2, _1>;

    // Build the epilogue
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape_MNK, ClusterShape_MNK,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementAccumulator, ElementC, LayoutC, AlignmentC, ElementC,
        LayoutC, AlignmentC,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

    // Build the mainloop
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB,
        LayoutB, AlignmentB, ElementAccumulator, MmaTileShape_MNK,
        ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

    using CollectiveAllReduce = cutlass::comm::collective::CollectiveAllReduceMulticastWarpSpecialized<
        ElementC, MmaTileShape_MNK, typename CollectiveEpilogue::StrideD>;

    // Compose into a kernel
    using GemmKernel = cutlass::gemm::kernel::Sm100GemmARUniversal<
        Shape<int, int, int, int>,  // Indicates ProblemShape
        CollectiveMainloop, CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler, CollectiveAllReduce>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Reference device GEMM implementation type
    using DeviceGemmReference =
        cutlass::reference::device::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                         ElementC, LayoutC,
                                         ElementAccumulator, ElementAccumulator>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    //
    // Data members
    //

    /// Initialization
    StrideA stride_A;
    StrideB stride_B;
    StrideC stride_C;
    StrideD stride_D;
    uint64_t seed = 1;

    cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
    cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
    cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
    nvshmemAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
    nvshmemAllocation<typename Gemm::EpilogueOutputOp::ElementOutput>
    block_D_red;
    nvshmemAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;
    cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D_red;

    __global__ void ref_reduce_kernel(ElementC *out, ElementC **ref_D_ptr,
                                      ElementC *arrD_red,
                                      ElementC *arrD, size_t npes, size_t nelem) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        volatile ElementC *output = out;
        volatile ElementC *val_ptr;
        for (int i = tid; i < nelem; i += gridDim.x * blockDim.x) {
            val_ptr = ref_D_ptr[0] + i;
            output[i] = *(val_ptr);
            for (int n = 1; n < npes; ++n) {
                val_ptr = ref_D_ptr[n] + i;
                output[i] += *(val_ptr);
            }
        }
    }

    __global__ void compare_kernel(ElementC *expected_out, ElementC *actual_out,
                                   ElementC **ref_D_ptr,
                                   ElementC *arrD_red, ElementC *arrD, int mype,
                                   size_t npes,
                                   size_t nelem) {

      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      for (int i = tid; i < nelem; i += gridDim.x * blockDim.x) {
        if (actual_out[i] != expected_out[i]) {
          printf("%d elem: %d, mismatch expected_out: %f, actual: %f
                  computed: %f : %f \n", mype, i, expected_out[i],
                  actual_out[i], *(ref_D_ptr[0] + i), *(ref_D_ptr[1] + i));
        }
      }
    }

    //////  nvshmem variables //////
    nvshmem_team_t *teams_dev, *teams;
    int num_teams;
    int mype, npes;

    /// Execute a given example GEMM computation
    template <typename Gemm>
    int run(Options &options) {
        initialize(options); // Initializes the inputs

        // Instantiate CUTLASS kernel depending on templates
        Gemm gemm;

        // Create a structure of gemm kernel arguments
        auto arguments = args_from_options(options);

        auto grid = gemm.get_grid_shape(arguments);
        dim3 blockShape = GemmKernel::get_block_shape();

        int sm_count;
        CUDA_CHECK(cudaDeviceGetAttribute(&sm_count,
                   cudaDevAttrMultiProcessorCount, 0));

        int max_active_blocks = gemm.maximum_active_blocks();
        printf("%d Grid dimension: (%d, %d, %d), block: (%d, %d, %d),
                occupancy: %d\n", mype, grid.x, grid.y, grid.z,
                blockShape.x, blockShape.y, blockShape.z, sm_count);

        int max_concurrent_blocks = sm_count * max_active_blocks;
        if (max_concurrent_blocks < (grid.x * grid.y * grid.z)) {
            fprintf(stderr,
              "Grid size exceeds maximum concurrent blocks. Using Tile-granular "
              "APIs requires all thread blocks to be concurrent across PEs\n");
            exit(1);
        }

        // create teams
        // each block has 1 warpgroup acting as epilogue, so num_teams = #blocks
        num_teams = grid.x * grid.y * grid.z;
        teams = (nvshmem_team_t *)malloc(num_teams * sizeof(nvshmem_team_t));

        for (int i = 0; i < num_teams; ++i) {
            nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 1, npes,
                                       nullptr, 0, &teams[i]);
        }
        CUDA_CHECK(cudaMalloc((void **)&teams_dev,
                              num_teams * sizeof(nvshmem_team_t)));
        CUDA_CHECK(cudaMemcpy(teams_dev, teams,
                   num_teams * sizeof(nvshmem_team_t), cudaMemcpyHostToDevice));

        // populate AR arguments
        arguments.allReduceArgs = {block_D.get(), block_D_red.get(), stride_D,
                                   nvshmem_my_pe(), nvshmem_n_pes(), teams_dev};

        // Using the arguments, query for extra workspace required
        // for matrix multiplication computation
        size_t workspace_size = Gemm::get_workspace_size(arguments);

        // Allocate workspace memory
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        // Check if the problem size is supported or not
        CUTLASS_CHECK(gemm.can_implement(arguments));

        // Initialize CUTLASS kernel with arguments and workspace pointer
        CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

        // Correctness / Warmup iteration
        CUTLASS_CHECK(gemm.run());

        // Check if output result and reference kernel are equal or not
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();
        Result result;
        result.passed = verify(options, mype, npes);

        std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed")
                  << std::endl;

        if (!result.passed) {
            exit(-1);
        }

        // Run profiling loop
        if (options.iterations > 0) {
            GpuTimer timer;
            timer.start();
            for (int iter = 0; iter < options.iterations; ++iter) {
                CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
                CUTLASS_CHECK(gemm.run());
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.stop();

            // Compute average runtime and GFLOPs.
            float elapsed_ms = timer.elapsed_millis();
            result.avg_runtime_ms = double(elapsed_ms) /
                                    double(options.iterations);
            result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

            std::cout << "  Problem Size: " << options.m << 'x'
                      << options.n << 'x' << options.k << std::endl;
            std::cout << "  Avg runtime: " << result.avg_runtime_ms
                      << " ms" << std::endl;
            std::cout << "  GFLOPS: " << result.gflops << std::endl;
        }

        return 0;
    }

    //////////////////////////////////////////////////////////////////

    template <class ElementT_, class TileShape_, class StrideMNL_>
    class CollectiveAllReduceMulticastWarpSpecialized {
       public:
        using ElementT = ElementT_;
        using TileShape = TileShape_;
        using StrideMNL = StrideMNL_;

        struct Arguments {
            ElementT* ptr_aux = nullptr;  // start pointer of matrix
            ElementT* out_ptr = nullptr;  // start pointer of matrix
            StrideMNL stride;
            int rank;
            int world_size;
            nvshmem_team_t* teams = nullptr;
        };

        struct Params {
            ElementT* ptr_aux = nullptr;
            ElementT* out_ptr = nullptr;
            StrideMNL stride;
            int rank;
            int world_size;
            Layout<Shape<int, int>> tile_layout;
            nvshmem_team_t* teams = nullptr;
        };

        template <class ProblemShape>
        static constexpr Params
        to_underlying_arguments(ProblemShape const& problem_shape,
                                Arguments const& args) {
            // Append 1s until problem shape is rank-4
            auto problem_shape_mnkl = append<4>(problem_shape, 1);
            auto [M, N, K, L] = problem_shape_mnkl;

            int m_tiles = ceil_div(M, size<0>(TileShape{}));
            int n_tiles = ceil_div(N, size<1>(TileShape{}));
            //  number of tiles in each dimension
            auto tile_layout = make_layout(make_shape(m_tiles, n_tiles));

            return {
                args.ptr_aux,    args.out_ptr, args.stride, args.rank,
                args.world_size, tile_layout,  args.teams,
            };
        }

        const Params* params_ptr;

        CUTLASS_HOST_DEVICE
        CollectiveAllReduceMulticastWarpSpecialized() {}

        CUTLASS_HOST_DEVICE
        CollectiveAllReduceMulticastWarpSpecialized(Params const& params):
                                                    params_ptr(&params) {}

        template <class ProblemShapeMNKL, class TileCoordMNKL>
        CUTLASS_DEVICE void do_allreduce(ProblemShapeMNKL const& problem_shape,
                                         TileCoordMNKL const& tile_coord) {
            auto [M, N, K, L] = problem_shape;
            auto [m, n, k, l] = tile_coord;

            if (m >= size<0>(params_ptr->tile_layout.shape()) ||
                n >= size<1>(params_ptr->tile_layout.shape())) {
                // early exit if out of bound
                return;
            }

            int tile_index = params_ptr->tile_layout(m, n);
            int tiles_per_rank =
                cute::ceil_div(cute::product(params_ptr->tile_layout.shape()),
                               params_ptr->world_size);

            // only root PE will do reduction for this tile
            // only needed if using two-shot algorithm
            int root = tile_index / tiles_per_rank;

            Tensor mAux = make_tensor(params_ptr->ptr_aux,
                                      make_layout(make_shape(M, N, L),
                                      params_ptr->stride));  // (M,N,L)
            Tensor mAux_out = make_tensor(
                params_ptr->out_ptr, make_layout(make_shape(M, N, L),
                params_ptr->stride));  // (M,N,L)

            Tensor gAux =
                local_tile(mAux, take<0, 2>(TileShape{}), make_coord(m, n, l));
            Tensor gAux_out =
                local_tile(mAux_out, take<0, 2>(TileShape{}),
                           make_coord(m, n, l));

            // predication tensor
            Tensor coordAux = make_identity_tensor(shape(mAux));
            Tensor pAux = local_tile(coordAux, take<0, 2>(TileShape{}),
                                     make_coord(m, n, l));

            auto boundary = nvshmemx::make_shape<int, int>(M, N);
            auto start_coord = nvshmemx::make_shape<int, int>(
                               size<0>(pAux(0, 0)), size<1>(pAux(0, 0)));

            // Call AR
            auto tensor_shape = nvshmemx::make_shape(M, N);
            auto tensor_stride = nvshmemx::make_stride(
                                      size<0>(params_ptr->stride),
                                      size<1>(params_ptr->stride));
            nvshmemx::Tensor srcTensor = nvshmemx::Tensor(gAux.data(),
                             nvshmemx::make_layout(tensor_shape, tensor_stride));
            nvshmemx::Tensor dstTensor = nvshmemx::Tensor(gAux_out.data(),
                             nvshmemx::make_layout(tensor_shape, tensor_stride));

            int blkId = blockIdx.x + gridDim.x * blockIdx.y;

            nvshmemx::tile_sum_allreduce_warpgroup<decltype(srcTensor),
                             decltype(dstTensor),
                             decltype(boundary),
                             nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI>(
                params_ptr->teams[blkId], srcTensor, dstTensor, start_coord,
                boundary, root, 0);
        }

        CUTLASS_DEVICE
        void tile_collective_wait() {
            int blkId = blockIdx.x + gridDim.x * blockIdx.y;
            nvshmemx::tile_collective_wait_warpgroup<
                nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PULL_NBI>(
            params_ptr->teams[blkId], 0);
        }
    };

    ///////////////////////////////////////////////////////////////////

    int main(int argc, char const **args) {
        // initialize nvshmem
        nvshmem_init();
        mype = nvshmem_my_pe();
        npes = nvshmem_n_pes();
        CUDA_CHECK(cudaSetDevice(mype));
        printf(" Executing PE: %d out of %d\n", mype, npes);

        // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
        // and must have compute capability at least 100a.

        if (__CUDACC_VER_MAJOR__ < 12 ||
           (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
            std::cerr << "This example requires CUDA 12.8 or newer."
                      << std::endl;
            // Returning zero so this test passes on older Toolkits.
            // Its actions are no-op.
            return 0;
        }

        cudaDeviceProp props;
        int current_device_id;
        CUDA_CHECK(cudaGetDevice(&current_device_id));
        CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
        cudaError_t error = cudaGetDeviceProperties(&props, 0);
        if (props.major != 10 || props.minor != 0) { std::cerr
            << "This example requires a GPU with compute capability 100a)."
            << std::endl;
            return 0;
        }

        //
        // Parse options
        //

        Options options;

        options.parse(argc, args);

        if (options.help) {
            options.print_usage(std::cout) << std::endl;
            return 0;
        }

        //
        // Evaluate CUTLASS kernels
        //

    #if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
        run<Gemm>(options);
    #endif  // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

        nvshmem_barrier_all();

        for (int i = 0; i < num_teams; ++i) {
            nvshmem_team_destroy(teams[i]);
        }
        nvshmem_barrier_all();

        block_D.free();
        block_D_red.free();
        block_ref_D.free();
        free(teams);
        CUDA_CHECK(cudaFree(teams_dev));
        nvshmem_finalize();
        return 0;
    }
