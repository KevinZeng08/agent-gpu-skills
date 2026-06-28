# Atomic Memory Operations

An AMO is a one-sided communication mechanism that combines memory read, update, or write operations with atomicity guarantees described in Section Atomicity Guarantees. Similar to the RMA routines, described in Section Remote Memory Access, the AMOs are performed only on symmetric objects.

Please note that AMOs are only supported as device operations for remote transports (i.e. ucx, ibrc). Host side AMOs are only supported for NVLink connected PEs.

NVSHMEM defines two types of AMO routines:

  * The _fetching_ routines return the original value of, and optionally update, the remote data object in a single atomic operation. The routines return after the data has been fetched from the target PE and delivered to the calling PE. The data type of the returned value is the same as the type of the remote data object.
  * The _non-fetching_ routines update the remote data object in a single atomic operation. A call to a non-fetching atomic routine issues the atomic operation and may return before the operation executes on the target PE. The `nvshmem_quiet`, `nvshmem_barrier`, or `nvshmem_barrier_all` routines can be used to force completion for these non-fetching atomic routines.

NVSHMEM provides AMO interfaces with the following types:

Standard AMO Types and Names _TYPE_ | _TYPENAME_
---|---
int | int
long | long
long long | longlong
size_t | size
ptrdiff_t | ptrdiff

Extended AMO Types and Names _TYPE_ | _TYPENAME_
---|---
float | float
double | double
int | int
long | long
long long | longlong
size_t | size
ptrdiff_t | ptrdiff

Bitwise AMO Types and Names _TYPE_ | _TYPENAME_
---|---
unsigned int | uint
unsigned long | ulong
unsigned long long | ulonglong
int32_t | int32
int64_t | int64
uint32_t | uint32
uint64_t | uint64

## **NVSHMEM_ATOMIC_FETCH**

`__device__ TYPE nvshmem_TYPENAME_atomic_fetch ( const TYPE *source , int pe )`

where _TYPE_ is one of the extended AMO types and has a corresponding _TYPENAME_ specified by Table extamotypes.

Symmetric address of the source data object. The type of `source` should match that implied in the SYNOPSIS section.

An integer that indicates the PE number from which `source` is to be fetched.

**Description**

`nvshmem_atomic_fetch` performs an atomic fetch operation. It returns the contents of the `source` as an atomic operation.

**Returns**

The contents at the `source` address on the remote PE. The data type of the return value is the same as the type of the remote data object.

## **NVSHMEM_ATOMIC_SET**

`__device__ void nvshmem_TYPENAME_atomic_set ( TYPE *dest , TYPE value , int pe )`

where _TYPE_ is one of the extended AMO types and has a corresponding _TYPENAME_ specified by Table extamotypes.

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

The operand to the atomic set operation. The type of `value` should match that implied in the SYNOPSIS section.

An integer that indicates the PE number on which `dest` is to be updated.

**Description**

`nvshmem_atomic_set` performs an atomic set operation. It writes the `value` into `dest` on `pe` as an atomic operation.

**Returns**

None.

## **NVSHMEM_ATOMIC_COMPARE_SWAP**

`__device__ TYPE nvshmem_TYPENAME_atomic_compare_swap ( TYPE *dest , TYPE cond , TYPE value , int pe )`

where _TYPE_ is one of the standard AMO types and has a corresponding _TYPENAME_ specified by Table stdamotypes.

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

`cond` is compared to the remote `dest` value. If `cond` and the remote `dest` are equal, then `value` is swapped into the remote `dest`; otherwise, the remote `dest` is unchanged. In either case, the old value of the remote `dest` is returned as the routine return value. `cond` must be of the same data type as `dest`.

The value to be atomically written to the remote PE. The type of `value` should match that implied in the SYNOPSIS section.

An integer that indicates the PE number upon which `dest` is to be updated.

**Description**

The conditional swap routines conditionally update a `dest` data object on the specified PE and return the prior contents of the data object in one atomic operation.

**Returns**

The contents that had been in the `dest` data object on the remote PE prior to the conditional swap. Data type is the same as the `dest` data type.

The following call ensures that the first PE to execute the conditional swap will successfully write its PE number to `race_winner` on PE `0`. ./example_code/shmem_atomic_compare_swap_example.c

## **NVSHMEM_ATOMIC_SWAP**

`__device__ TYPE nvshmem_TYPENAME_atomic_swap ( TYPE *dest , TYPE value , int pe )`

where _TYPE_ is one of the extended AMO types and has a corresponding _TYPENAME_ specified by Table extamotypes.

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

The value to be atomically written to the remote PE. The type of `value` should match that implied in the SYNOPSIS section.

An integer that indicates the PE number on which `dest` is to be updated.

**Description**

`nvshmem_atomic_swap` performs an atomic swap operation. It writes `value` into `dest` on PE and returns the previous contents of `dest` as an atomic operation.

**Returns**

The content that had been at the `dest` address on the remote PE prior to the swap is returned.

The example below swaps values between odd numbered PEs and their right (modulo) neighbor and outputs the result of swap. ./example_code/shmem_atomic_swap_example.c

## **NVSHMEM_ATOMIC_FETCH_INC**

`__device__ TYPE nvshmem_TYPENAME_atomic_fetch_inc ( TYPE *dest , int pe )`

where _TYPE_ is one of the standard AMO types and has a corresponding _TYPENAME_ specified by Table stdamotypes.

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

An integer that indicates the PE number on which `dest` is to be updated.

**Description**

These routines perform a fetch-and-increment operation. The `dest` on PE `pe` is increased by one and the routine returns the previous contents of `dest` as an atomic operation.

**Returns**

The contents that had been at the `dest` address on the remote PE prior to the increment. The data type of the return value is the same as the `dest`.

The following `nvshmem_atomic_fetch_inc` example is for _C_ programs: ./example_code/shmem_atomic_fetch_inc_example.c

## **NVSHMEM_ATOMIC_INC**

`__device__ void nvshmem_TYPENAME_atomic_inc ( TYPE *dest , int pe )`

where _TYPE_ is one of the standard AMO types and has a corresponding _TYPENAME_ specified by Table stdamotypes.

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

An integer that indicates the PE number on which `dest` is to be updated.

**Description**

These routines perform an atomic increment operation on the `dest` data object on PE.

**Returns**

None.

The following `nvshmem_atomic_inc` example is for _C_ programs: ./example_code/shmem_atomic_inc_example.c

## **NVSHMEM_ATOMIC_FETCH_ADD**

`__device__ TYPE nvshmem_TYPENAME_atomic_fetch_add ( TYPE *dest , TYPE value , int pe )`

where _TYPE_ is one of the standard AMO types and has a corresponding _TYPENAME_ specified by Table stdamotypes.

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

The operand to the atomic fetch-and-add operation. The type of `value` should match that implied in the SYNOPSIS section.

An integer that indicates the PE number on which `dest` is to be updated.

**Description**

`nvshmem_atomic_fetch_add` routines perform an atomic fetch-and-add operation. An atomic fetch-and-add operation fetches the old `dest` and adds `value` to `dest` without the possibility of another atomic operation on the `dest` between the time of the fetch and the update. These routines add `value` to `dest` on `pe` and return the previous contents of `dest` as an atomic operation.

**Returns**

The contents that had been at the `dest` address on the remote PE prior to the atomic addition operation. The data type of the return value is the same as the `dest`.

The following `nvshmem_atomic_fetch_add` example is for _C_ programs: ./example_code/shmem_atomic_fetch_add_example.c

## **NVSHMEM_ATOMIC_ADD**

`__device__ void nvshmem_TYPENAME_atomic_add ( TYPE *dest , TYPE value , int pe )`

where _TYPE_ is one of the standard AMO types and has a corresponding _TYPENAME_ specified by Table stdamotypes.

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

The operand to the atomic add operation. The type of `value` should match that implied in the SYNOPSIS section.

An integer that indicates the PE number upon which `dest` is to be updated.

**Description**

The `nvshmem_atomic_add` routine performs an atomic add operation. It adds `value` to `dest` on PE `pe` and atomically updates the `dest` without returning the value.

**Returns**

None.

./example_code/shmem_atomic_add_example.c

## **NVSHMEM_ATOMIC_FETCH_AND**

`__device__ TYPE nvshmem_TYPENAME_atomic_fetch_and ( TYPE *dest , TYPE value , int pe )`

where _TYPE_ is one of the bitwise AMO types and has a corresponding _TYPENAME_ specified by Table bitamotypes.

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

The operand to the bitwise AND operation. The type of `value` should match that implied in the SYNOPSIS section.

An integer value for the PE on which `dest` is to be updated.

**Description**

`nvshmem_atomic_fetch_and` atomically performs a fetching bitwise AND on the remotely accessible data object pointed to by `dest` at PE `pe` with the operand `value`.

**Returns**

The value pointed to by `dest` on PE `pe` immediately before the operation is performed.

## **NVSHMEM_ATOMIC_AND**

`__device__ void nvshmem_TYPENAME_atomic_and ( TYPE *dest , TYPE value , int pe )`

where _TYPE_ is one of the bitwise AMO types and has a corresponding _TYPENAME_ specified by Table bitamotypes.

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

The operand to the bitwise AND operation. The type of `value` should match that implied in the SYNOPSIS section.

An integer value for the PE on which `dest` is to be updated.

**Description**

`nvshmem_atomic_and` atomically performs a non-fetching bitwise AND on the remotely accessible data object pointed to by `dest` at PE `pe` with the operand `value`.

**Returns**

None.

## **NVSHMEM_ATOMIC_FETCH_OR**

`__device__ TYPE nvshmem_TYPENAME_atomic_fetch_or ( TYPE *dest , TYPE value , int pe )`

where _TYPE_ is one of the bitwise AMO types and has a corresponding _TYPENAME_ specified by Table bitamotypes.

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

The operand to the bitwise OR operation. The type of `value` should match that implied in the SYNOPSIS section.

An integer value for the PE on which `dest` is to be updated.

**Description**

`nvshmem_atomic_fetch_or` atomically performs a fetching bitwise OR on the remotely accessible data object pointed to by `dest` at PE `pe` with the operand `value`.

**Returns**

The value pointed to by `dest` on PE `pe` immediately before the operation is performed.

## **NVSHMEM_ATOMIC_OR**

`__device__ void nvshmem_TYPENAME_atomic_or ( TYPE *dest , TYPE value , int pe )`

where _TYPE_ is one of the bitwise AMO types and has a corresponding _TYPENAME_ specified by Table bitamotypes.

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

The operand to the bitwise OR operation. The type of `value` should match that implied in the SYNOPSIS section.

An integer value for the PE on which `dest` is to be updated.

**Description**

`nvshmem_atomic_or` atomically performs a non-fetching bitwise OR on the remotely accessible data object pointed to by `dest` at PE `pe` with the operand `value`.

**Returns**

None.

## **NVSHMEM_ATOMIC_FETCH_XOR**

`__device__ TYPE nvshmem_TYPENAME_atomic_fetch_xor ( TYPE *dest , TYPE value , int pe )`

where _TYPE_ is one of the bitwise AMO types and has a corresponding _TYPENAME_ specified by Table bitamotypes.

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

The operand to the bitwise XOR operation. The type of `value` should match that implied in the SYNOPSIS section.

An integer value for the PE on which `dest` is to be updated.

**Description**

`nvshmem_atomic_fetch_xor` atomically performs a fetching bitwise XOR on the remotely accessible data object pointed to by `dest` at PE `pe` with the operand `value`.

**Returns**

The value pointed to by `dest` on PE `pe` immediately before the operation is performed.

## **NVSHMEM_ATOMIC_XOR**

`__device__ void nvshmem_TYPENAME_atomic_xor ( TYPE *dest , TYPE value , int pe )`

where _TYPE_ is one of the bitwise AMO types and has a corresponding _TYPENAME_ specified by Table bitamotypes.

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

The operand to the bitwise XOR operation. The type of `value` should match that implied in the SYNOPSIS section.

An integer value for the PE on which `dest` is to be updated.

**Description**

`nvshmem_atomic_xor` atomically performs a non-fetching bitwise XOR on the remotely accessible data object pointed to by `dest` at PE `pe` with the operand `value`.

**Returns**

None.
