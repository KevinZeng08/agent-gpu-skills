# Collective Communication Functions

The following NCCL APIs provide some commonly used collective operations.

## ncclAllReduce

### [ncclResult_t](types.md#c.ncclResult_t) ncclAllReduce(const void \*sendbuff, void \*recvbuff, size_t count, [ncclDataType_t](types.md#c.ncclDataType_t) datatype, [ncclRedOp_t](types.md#c.ncclRedOp_t) op, [ncclComm_t](types.md#c.ncclComm_t) comm, cudaStream_t stream)

Reduces data arrays of length `count` in `sendbuff` using the `op` operation and leaves identical copies of the result in each `recvbuff`.

In-place operation will happen if `sendbuff == recvbuff`.

Related links: [AllReduce](../usage/collectives.md#allreduce).

## ncclBroadcast

### [ncclResult_t](types.md#c.ncclResult_t) ncclBroadcast(const void \*sendbuff, void \*recvbuff, size_t count, [ncclDataType_t](types.md#c.ncclDataType_t) datatype, int root, [ncclComm_t](types.md#c.ncclComm_t) comm, cudaStream_t stream)

Copies `count` elements from `sendbuff` on the `root` rank to all ranks’ `recvbuff`.
`sendbuff` is only used on rank `root` and ignored for other ranks.

In-place operation will happen if `sendbuff == recvbuff`.

### [ncclResult_t](types.md#c.ncclResult_t) ncclBcast(void \*buff, size_t count, [ncclDataType_t](types.md#c.ncclDataType_t) datatype, int root, [ncclComm_t](types.md#c.ncclComm_t) comm, cudaStream_t stream)

Legacy in-place version of `ncclBroadcast` in a similar fashion to MPI_Bcast. A call to

```c++
ncclBcast(buff, count, datatype, root, comm, stream)
```

is equivalent to

```c++
ncclBroadcast(buff, buff, count, datatype, root, comm, stream)
```

Related links: [Broadcast](../usage/collectives.md#broadcast)

## ncclReduce

### [ncclResult_t](types.md#c.ncclResult_t) ncclReduce(const void \*sendbuff, void \*recvbuff, size_t count, [ncclDataType_t](types.md#c.ncclDataType_t) datatype, [ncclRedOp_t](types.md#c.ncclRedOp_t) op, int root, [ncclComm_t](types.md#c.ncclComm_t) comm, cudaStream_t stream)

Reduce data arrays of length `count` in `sendbuff` into `recvbuff` on the `root` rank using the `op` operation.
`recvbuff` is only used on rank `root` and ignored for other ranks.

In-place operation will happen if `sendbuff == recvbuff`.

Related links: [Reduce](../usage/collectives.md#reduce).

## ncclAllGather

### [ncclResult_t](types.md#c.ncclResult_t) ncclAllGather(const void \*sendbuff, void \*recvbuff, size_t sendcount, [ncclDataType_t](types.md#c.ncclDataType_t) datatype, [ncclComm_t](types.md#c.ncclComm_t) comm, cudaStream_t stream)

Gathers `sendcount` values from all GPUs and leaves identical copies of the result in each `recvbuff`, receiving data from rank `i` at offset `i*sendcount`.

Note: This assumes the receive count is equal to `nranks*sendcount`, which means that `recvbuff` should have a size of at least `nranks*sendcount` elements.

In-place operation will happen if `sendbuff == recvbuff + rank * sendcount`.

Related links: [AllGather](../usage/collectives.md#allgather), [In-place Operations](../usage/inplace.md#in-place-operations).

## ncclReduceScatter

### [ncclResult_t](types.md#c.ncclResult_t) ncclReduceScatter(const void \*sendbuff, void \*recvbuff, size_t recvcount, [ncclDataType_t](types.md#c.ncclDataType_t) datatype, [ncclRedOp_t](types.md#c.ncclRedOp_t) op, [ncclComm_t](types.md#c.ncclComm_t) comm, cudaStream_t stream)

Reduce data in `sendbuff` from all GPUs using the `op` operation and leave the reduced result scattered over the devices so that the `recvbuff` on
rank `i` will contain the i-th block of the result.

Note:  This assumes the send count is equal to `nranks*recvcount`, which means that `sendbuff` should have a size of at least `nranks*recvcount` elements.

In-place operation will happen if `recvbuff == sendbuff + rank * recvcount`.

Related links: [ReduceScatter](../usage/collectives.md#reducescatter), [In-place Operations](../usage/inplace.md#in-place-operations).

## ncclAlltoAll

### [ncclResult_t](types.md#c.ncclResult_t) ncclAlltoAll(const void \*sendbuff, void \*recvbuff, size_t count, [ncclDataType_t](types.md#c.ncclDataType_t) datatype, [ncclComm_t](types.md#c.ncclComm_t) comm, cudaStream_t stream)

Each rank sends `count` values to all other ranks and receives `count` values from all other ranks. Data to send to destination rank `j` is taken from `sendbuff+j*count` and data received from source rank `i` is placed at `recvbuff+i*count`.

Note: This assumes both the total send and receive count is equal to `nranks*count`, which means that `sendbuff` and `recvbuff` should have a size of at least `nranks*count` elements.

In-place operation is currently not supported.

Related links: [AlltoAll](../usage/collectives.md#alltoall).

## ncclGather

### [ncclResult_t](types.md#c.ncclResult_t) ncclGather(const void \*sendbuff, void \*recvbuff, size_t count, [ncclDataType_t](types.md#c.ncclDataType_t) datatype, int root, [ncclComm_t](types.md#c.ncclComm_t) comm, cudaStream_t stream)

Each rank sends `count` elements from `sendbuff` to the `root` rank. On the `root` rank, data from rank `i` is placed at `recvbuff + i*count`. On non-root ranks, `recvbuff` is not used.

Note: This assumes the receive count is equal to `nranks*count`, which means that `recvbuff` should have a size of at least `nranks*count` elements.

In-place operation will happen if `sendbuff == recvbuff + root * count`.

Related links: [Gather](../usage/collectives.md#gather).

## ncclScatter

### [ncclResult_t](types.md#c.ncclResult_t) ncclScatter(const void \*sendbuff, void \*recvbuff, size_t count, [ncclDataType_t](types.md#c.ncclDataType_t) datatype, int root, [ncclComm_t](types.md#c.ncclComm_t) comm, cudaStream_t stream)

Each rank receives `count` elements from the `root` rank. On the `root` rank, `count` elements from `sendbuff + i*count` are sent to rank `i`. On non-root ranks, `sendbuff` is not used.

Note: This assumes the send count is equal to `nranks*count`, which means that `sendbuff` should have a size of at least `nranks*count` elements.

In-place operation will happen if `recvbuff == sendbuff + root * count`.

Related links: [Scatter](../usage/collectives.md#scatter).
