# Remote Memory Access

The RMA routines described in this section can be used to perform reads from and writes to symmetric data objects. These operations are one-sided, meaning that the PE invoking an operation provides all communication parameters and the targeted PE is passive. A characteristic of one-sided communication is that it decouples communication from synchronization. One-sided communication mechanisms transfer data; however, they do not synchronize the sender of the data with the receiver of the data.

NVSHMEM’s RMA routines are performed on symmetric data objects. The initiator PE of a call is designated as the _origin_ PE and the PE targeted by an operation is designated as the _destination_ PE. The `source` and `dest` designators refer to the data objects that an operation reads from and writes to. In the case of the remote update routine, _Put_ , the origin PE provides the `source` data object and the destination PE provides the `dest` data object. In the case of the remote read routine, _Get_ , the origin PE provides the `dest` data object and the destination PE provides the `source` data object.

The standard RMA types include the exact-width integer types defined in `stdint.h` by _C_ §7.18.1.1 and _C_ §7.20.1.1. When the _C_ translation environment does not provide exact-width integer types with `stdint.h`, an NVSHMEM implemementation is not required to provide support for these types.

Similar to tile-granular collectives, NVSHMEM supports tile-granular RMA routines (tile_put and tile_get). which take tiles as input and output. Unlike tile-granular collectives, tile-granular RMA routines are not restricted to NVLink SHARP based systems and can be used on systems supported by NVSHMEM. The users are expected to specify the algorithm to be used. The list of algorithms supported is shown below:

  * `tile_algo_t::PEER_PUSH_NBI`
  * `tile_algo_t::PEER_PULL_NBI`
  * `tile_algo_t::REMOTE_PUSH_NBI`
  * `tile_algo_t::REMOTE_PULL_NBI`

Standard RMA Types and Names _TYPE_ | _TYPENAME_
---|---
float | float
double | double
__nv_bfloat16 | bfloat16
half | half
char | char
signed char | schar
short | short
int | int
long | long
long long | longlong
unsigned char | uchar
unsigned short | ushort
unsigned int | uint
unsigned long | ulong
unsigned long long | ulonglong
int8_t | int8
int16_t | int16
int32_t | int32
int64_t | int64
uint8_t | uint8
uint16_t | uint16
uint32_t | uint32
uint64_t | uint64
size_t | size
ptrdiff_t | ptrdiff

## Blocking RMA

### **NVSHMEM_PUT**

`void nvshmem_TYPENAME_put ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

`void nvshmemx_TYPENAME_put_on_stream ( TYPE *dest , const TYPE *source , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_TYPENAME_put ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

`__device__ void nvshmemx_TYPENAME_put_block ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

`__device__ void nvshmemx_TYPENAME_put_warp ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

`void nvshmem_putSIZE ( void *dest , const void *source , size_t nelems , int pe )`

`void nvshmemx_putSIZE_on_stream ( void *dest , const void *source , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_putSIZE ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_putSIZE_block ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_putSIZE_warp ( void *dest , const void *source , size_t nelems , int pe )`

where _SIZE_ is one of `8, 16, 32, 64, 128`.

`void nvshmem_putmem ( void *dest , const void *source , size_t nelems , int pe )`

`void nvshmemx_putmem_on_stream ( void *dest , const void *source , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_putmem ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_putmem_block ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_putmem_warp ( void *dest , const void *source , size_t nelems , int pe )`

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

Symmetric address or host/device address registered via `nvshmemx_buffer_register` of the data object containing the data to be copied. The type of `source` should match that implied in the SYNOPSIS section.Additionally, it can also be backed by device shared memory when devices are connected via peer-to-peer transport.

Number of elements in the `dest` and `source` arrays. For `nvshmem_putmem`, elements are bytes.

PE number of the remote PE.

**Description**

The routines return after the data has been copied out of the `source` array on the local PE. The delivery of data words into the data object on the destination PE may occur in any order. Furthermore, two successive put routines may deliver data out of order unless a call to `nvshmem_fence` is introduced between the two calls.

**Returns**

None.

The following `nvshmem_put` example is for _C_ programs: ./example_code/shmem_put_example.c

### **NVSHMEM_P**

`void nvshmem_TYPENAME_p ( TYPE *dest , TYPE value , int pe )`

`__device__ void nvshmem_TYPENAME_p ( TYPE *dest , TYPE value , int pe )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

The value to be transferred to `dest`. The type of `value` should match that implied in the SYNOPSIS section.

The number of the remote PE.

**Description**

These routines provide a very low latency put capability for single elements of most basic types.

As with `nvshmem_put`, these routines start the remote transfer and may return before the data is delivered to the remote PE. Use `nvshmem_quiet` to force completion of all remote _Put_ transfers.

**Returns**

None.

The following example uses `nvshmem_p` in a _C_ program. ./example_code/shmem_p_example.c

### **NVSHMEM_IPUT**

`void nvshmem_TYPENAME_iput ( TYPE *dest , const TYPE *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

`void nvshmemx_TYPENAME_iput_on_stream ( TYPE *dest , const TYPE *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_TYPENAME_iput ( TYPE *dest , const TYPE *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

`__device__ void nvshmemx_TYPENAME_iput_block ( TYPE *dest , const TYPE *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

`__device__ void nvshmemx_TYPENAME_iput_warp ( TYPE *dest , const TYPE *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

`void nvshmem_iputSIZE ( void *dest , const void *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

`void nvshmemx_iputSIZE_on_stream ( void *dest , const void *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_iputSIZE ( void *dest , const void *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

`__device__ void nvshmemx_iputSIZE_block ( void *dest , const void *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

`__device__ void nvshmemx_iputSIZE_warp ( void *dest , const void *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

where _SIZE_ is one of `8, 16, 32, 64, 128`.

Symmetric address of the destination array data object. The type of `dest` should match that implied in the SYNOPSIS section.

Symmetric address or host/device address registered via `nvshmemx_buffer_register` of the array containing the data to be copied. The type of `source` should match that implied in the SYNOPSIS section.Additionally, it can also be backed by device shared memory when devices are connected via peer-to-peer transport.

The stride between consecutive elements of the `dest` array. The stride is scaled by the element size of the `dest` array. A value of `1` indicates contiguous data.

The stride between consecutive elements of the `source` array. The stride is scaled by the element size of the `source` array. A value of `1` indicates contiguous data.

Number of elements in the `dest` and `source` arrays.

PE number of the remote PE.

**Description**

The `iput` routines provide a method for copying strided data elements (specified by `sst`) of an array from a `source` array on the local PE to locations specified by stride `dst` on a `dest` array on specified remote PE. Both strides, `dst` and `sst`, must be greater than or equal to `1`. The routines return when the data has been copied out of the `source` array on the local PE but not necessarily before the data has been delivered to the remote data object.

**Returns**

None.

**Notes**

See Section Memory Model for a definition of the term remotely accessible.

Consider the following `nvshmem_iput` example for _C_ programs. ./example_code/shmem_iput_example.c

### **NVSHMEM_GET**

`void nvshmem_TYPENAME_get ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

`void nvshmemx_TYPENAME_get_on_stream ( TYPE *dest , const TYPE *source , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_TYPENAME_get ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

`__device__ void nvshmemx_TYPENAME_get_block ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

`__device__ void nvshmemx_TYPENAME_get_warp ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

`void nvshmem_getSIZE ( void *dest , const void *source , size_t nelems , int pe )`

`void nvshmemx_getSIZE_on_stream ( void *dest , const void *source , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_getSIZE ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_getSIZE_block ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_getSIZE_warp ( void *dest , const void *source , size_t nelems , int pe )`

where _SIZE_ is one of `8, 16, 32, 64, 128`.

`void nvshmem_getmem ( void *dest , const void *source , size_t nelems , int pe )`

`void nvshmemx_getmem_on_stream ( void *dest , const void *source , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_getmem ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_getmem_block ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_getmem_warp ( void *dest , const void *source , size_t nelems , int pe )`

Symmetric address or host/device address registered via `nvshmemx_buffer_register` of the data object to be updated. The type of `dest` should match that implied in the SYNOPSIS section.Additionally, it can also be backed by device shared memory when devices are connected via peer-to-peer transport.

Symmetric address of the source data object. The type of `source` should match that implied in the SYNOPSIS section.

Number of elements in the `dest` and `source` arrays. For `nvshmem_getmem`, elements are bytes.

PE number of the remote PE.

**Description**

The get routines provide a method for copying a contiguous symmetric data object from a different PE to a contiguous data object on the local PE. The routines return after the data has been delivered to the `dest` array on the local PE.

**Returns**

None.

**Notes**

See Section Memory Model for a definition of the term remotely accessible.

### **NVSHMEM_G**

`TYPE nvshmem_TYPENAME_g ( const TYPE *source , int pe )`

`__device__ TYPE nvshmem_TYPENAME_g ( const TYPE *source , int pe )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

Symmetric address of the source data object. The type of `source` should match that implied in the SYNOPSIS section.

The number of the remote PE on which `source` resides.

**Description**

These routines provide a very low latency get capability for single elements of most basic types.

**Returns**

Returns a single element of type specified in the synopsis.

The following `nvshmem_g` example is for _C_ programs: ./example_code/shmem_g_example.c

### **NVSHMEM_IGET**

`void nvshmem_TYPENAME_iget ( TYPE *dest , const TYPE *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

`void nvshmemx_TYPENAME_iget_on_stream ( TYPE *dest , const TYPE *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_TYPENAME_iget ( TYPE *dest , const TYPE *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

`__device__ void nvshmemx_TYPENAME_iget_block ( TYPE *dest , const TYPE *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

`__device__ void nvshmemx_TYPENAME_iget_warp ( TYPE *dest , const TYPE *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

`void nvshmem_igetSIZE ( void *dest , const void *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

`void nvshmemx_igetSIZE_on_stream ( void *dest , const void *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_igetSIZE ( void *dest , const void *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

`__device__ void nvshmemx_igetSIZE_block ( void *dest , const void *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

`__device__ void nvshmemx_igetSIZE_warp ( void *dest , const void *source , ptrdiff_t dst , ptrdiff_t sst , size_t nelems , int pe )`

where _SIZE_ is one of `8, 16, 32, 64, 128`.

Symmetric address or host/device address registered via `nvshmemx_buffer_register` of the array to be updated. The type of `dest` should match that implied in the SYNOPSIS section.Additionally, it can also be backed by device shared memory when devices are connected via peer-to-peer transport.

Symmetric address of the source array data object. The type of `source` should match that implied in the SYNOPSIS section.

The stride between consecutive elements of the `dest` array. The stride is scaled by the element size of the `dest` array. A value of `1` indicates contiguous data.

The stride between consecutive elements of the `source` array. The stride is scaled by the element size of the `source` array. A value of `1` indicates contiguous data.

Number of elements in the `dest` and `source` arrays.

PE number of the remote PE.

**Description**

The `iget` routines provide a method for copying strided data elements from a symmetric array from a specified remote PE to strided locations on a local array. The routines return when the data has been copied into the local `dest` array.

**Returns**

None.

## Nonblocking RMA

### **NVSHMEM_PUT_NBI**

`void nvshmem_TYPENAME_put_nbi ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

`void nvshmemx_TYPENAME_put_nbi_on_stream ( TYPE *dest , const TYPE *source , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_TYPENAME_put_nbi ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

`__device__ void nvshmemx_TYPENAME_put_nbi_block ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

`__device__ void nvshmemx_TYPENAME_put_nbi_warp ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

`void nvshmem_putSIZE_nbi ( void *dest , const void *source , size_t nelems , int pe )`

`void nvshmemx_putSIZE_nbi_on_stream ( void *dest , const void *source , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_putSIZE_nbi ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_putSIZE_nbi_block ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_putSIZE_nbi_warp ( void *dest , const void *source , size_t nelems , int pe )`

where _SIZE_ is one of `8, 16, 32, 64, 128`.

`void nvshmem_putmem_nbi ( void *dest , const void *source , size_t nelems , int pe )`

`void nvshmemx_putmem_nbi_on_stream ( void *dest , const void *source , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_putmem_nbi ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_putmem_nbi_block ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_putmem_nbi_warp ( void *dest , const void *source , size_t nelems , int pe )`

Symmetric address of the destination data object. The type of `dest` should match that implied in the SYNOPSIS section.

Symmetric address or host/device address registered via `nvshmemx_buffer_register` of the object containing the data to be copied. The type of `source` should match that implied in the SYNOPSIS section.Additionally, it can also be backed by device shared memory when devices are connected via peer-to-peer transport.

Number of elements in the `dest` and `source` arrays. For `nvshmem_putmem_nbi` and `nvshmem_ctx_putmem_nbi`, elements are bytes.

PE number of the remote PE.

**Description**

The routines return after initiating the operation. The operation is considered complete after a subsequent call to `nvshmem_quiet`. At the completion of `nvshmem_quiet`, the data has been copied into the `dest` array on the destination PE. The delivery of data words into the data object on the destination PE may occur in any order. Furthermore, two successive put routines may deliver data out of order unless a call to `nvshmem_fence` is introduced between the two calls.

**Returns**

None.

### **NVSHMEM_GET_NBI**

`void nvshmem_TYPENAME_get_nbi ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

`void nvshmemx_TYPENAME_get_nbi_on_stream ( TYPE *dest , const TYPE *source , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_TYPENAME_get_nbi ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

`__device__ void nvshmemx_TYPENAME_get_nbi_block ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

`__device__ void nvshmemx_TYPENAME_get_nbi_warp ( TYPE *dest , const TYPE *source , size_t nelems , int pe )`

where _TYPE_ is one of the standard RMA types and has a corresponding _TYPENAME_ specified by Table stdrmatypes.

`void nvshmem_getSIZE_nbi ( void *dest , const void *source , size_t nelems , int pe )`

`void nvshmemx_getSIZE_nbi_on_stream ( void *dest , const void *source , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_getSIZE_nbi ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_getSIZE_nbi_block ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_getSIZE_nbi_warp ( void *dest , const void *source , size_t nelems , int pe )`

where _SIZE_ is one of `8, 16, 32, 64, 128`.

`void nvshmem_getmem_nbi ( void *dest , const void *source , size_t nelems , int pe )`

`void nvshmemx_getmem_nbi_on_stream ( void *dest , const void *source , size_t nelems , int pe , cudaStream_t stream )`

`__device__ void nvshmem_getmem_nbi ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_getmem_nbi_block ( void *dest , const void *source , size_t nelems , int pe )`

`__device__ void nvshmemx_getmem_nbi_warp ( void *dest , const void *source , size_t nelems , int pe )`

Symmetric address or host/device address registered via `nvshmemx_buffer_register` of the data object to be updated. The type of `dest` should match that implied in the SYNOPSIS section.Additionally, it can also be backed by device shared memory when devices are connected via peer-to-peer transport.

Symmetric address of the source data object. The type of `source` should match that implied in the SYNOPSIS section.

Number of elements in the `dest` and `source` arrays. For `nvshmem_getmem_nbi` and `nvshmem_ctx_getmem_nbi`, elements are bytes.

PE number of the remote PE.

**Description**

The get routines provide a method for copying a contiguous symmetric data object from a different PE to a contiguous data object on the local PE. The routines return after initiating the operation. The operation is considered complete after a subsequent call to `nvshmem_quiet`. At the completion of `nvshmem_quiet`, the data has been delivered to the `dest` array on the local PE.

**Returns**

None.

**Notes**

See Section Memory Model for a definition of the term remotely accessible.

## Tile RMA

### **TILE_PUT**

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_algo_t algo > __device__ int tile_put ( src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int pe , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_algo_t algo > __device__ int tile_put_warp ( src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int pe , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_algo_t algo > __device__ int tile_put_warpgroup ( src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int pe , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_algo_t algo > __device__ int tile_put_block ( src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int pe , uint64_t flags )`

_src [IN]_
    Tensor corresponding to the source data tile for the collective.
_dst [OUT]_
    Tensor corresponding to the destination data tile of the collective.
_start_coord [IN]_
    Tuple (of type tuple_t) containing the coordinate of the starting element of the src tensor.
_boundary [IN]_
    Tuple (of type tuple_t) containing the actual problem size of the multi-dimensional array / matrix of the src tensor.
_pe [IN]_
    PE number of remote PE.
_flags [IN]_
    Flags to be passed into the API call. Currently unused and should be set to 0.

**Description**

> This function performs a PUT operation to send an individual tile of data to a remote PE. The src tensor is local, while the dst tensor resides on the remote PE (specified by pe). Both tensors are of type src_tensor_t and dst_tensor_t respectively, created using Tile helper functions (make_layout(), Tensor()), which contain layout and datatype information.
>
> Currently supported data types are float, half, cutlass::half_t, __nv_bfloat16, and cutlass::bfloat16_t. Two algorithms are available: tile_algo_t::PEER_PUSH_NBI for NVLink P2P destinations and tile_algo_t::REMOTE_PUSH_NBI for remote destinations.
>
> When the problem size does not exactly match the tile size, out-of-bounds (OOB) accesses can be predicated using start_coord and boundary. The size of these tuples must equal the number of dimensions in the src tensor. Use the make_shape() helper function to create these tuples.

**Returns**
    On success, the function returns zero; otherwise, a non-zero error code is returned.

### **TILE_GET**

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_algo_t algo > __device__ int tile_get ( src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int pe , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_algo_t algo > __device__ int tile_get_warpgroup ( src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int pe , uint64_t flags )`

`template < typename src_tensor_t, typename dst_tensor_t, typename tuple_t, nvshmemx::tile_algo_t algo > __device__ int tile_get_block ( src_tensor_t src , dst_tensor_t dst , tuple_t start_coord , tuple_t boundary , int pe , uint64_t flags )`

_src [IN]_
    Tensor corresponding to the source data tile for the collective.
_dst [OUT]_
    Tensor corresponding to the destination data tile of the collective.
_start_coord [IN]_
    Tuple (of type tuple_t) containing the coordinate of the starting element of the src tensor.
_boundary [IN]_
    Tuple (of type tuple_t) containing the actual problem size of the multi-dimensional array / matrix of the src tensor.
_pe [IN]_
    PE number of remote PE.
_flags [IN]_
    Flags to be passed into the API call. Currently unused and should be set to 0.

**Description**

> This function performs a GET operation to retrieve an individual tile of data from a remote PE. The src tensor resides on the remote PE (specified by pe), while the dst tensor is local. Both tensors are of type src_tensor_t and dst_tensor_t respectively, created using Tile helper functions (make_layout(), Tensor()), which contain layout and datatype information.
>
> Currently supported data types are float, half, cutlass::half_t, __nv_bfloat16, and cutlass::bfloat16_t. Two algorithms are available: tile_algo_t::PEER_PULL_NBI for NVLink P2P destinations and tile_algo_t::REMOTE_PULL_NBI for remote destinations.
>
> When the problem size does not exactly match the tile size, out-of-bounds (OOB) accesses can be predicated using start_coord and boundary. The size of these tuples must equal the number of dimensions in the src tensor. Use the make_shape() helper function to create these tuples.

**Returns**
    On success, the function returns zero; otherwise, a non-zero error code is returned.
