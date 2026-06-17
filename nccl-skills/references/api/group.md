# Group Calls

Group primitives define the behavior of the current thread to avoid blocking. They can therefore be used from multiple threads independently.

Related links: [Group Calls](../usage/groups.md#group-calls).

## ncclGroupStart

### [ncclResult_t](types.md#c.ncclResult_t) ncclGroupStart()

Start a group call.

All subsequent calls to NCCL until ncclGroupEnd will not block due to inter-CPU synchronization.

## ncclGroupEnd

### [ncclResult_t](types.md#c.ncclResult_t) ncclGroupEnd()

End a group call.

Returns when all operations since ncclGroupStart have been processed. This means the communication primitives
have been enqueued to the provided streams, but are not necessarily complete.

When used with the ncclCommInitRank call, the ncclGroupEnd call waits for all communicators to be initialized.

## ncclGroupSimulateEnd

### [ncclResult_t](types.md#c.ncclResult_t) ncclGroupSimulateEnd([ncclSimInfo_t](types.md#c.ncclSimInfo_t) \*simInfo)

Simulate a ncclGroupEnd() call and return NCCL’s simulation info in a structure passed as an argument.
