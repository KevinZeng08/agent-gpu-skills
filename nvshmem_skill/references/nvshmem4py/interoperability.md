# Interoperability

NVSHMEM4Py provides interoperability with other Python libraries through the use of DLPack-compatible CUDA buffers. This allows for seamless integration with libraries like CuPy and PyTorch. It has special functions which support allocating CuPy Arrays and Torch tensors which are backed by NVSHMEM symmetric memory.

## Interoperability with PyTorch and CuPy

NVSHMEM4Py provides specialized functions to create PyTorch tensors and CuPy arrays that are backed by NVSHMEM symmetric memory. This allows these objects to be used directly in NVSHMEM operations while maintaining their native API functionality.

### Creating NVSHMEM-backed Arrays

The interoperability modules provide factory functions that create arrays backed by NVSHMEM symmetric memory:

    import nvshmem.core as nvshmem
    import torch
    import cupy as cp

    # Create a PyTorch tensor backed by NVSHMEM memory
    tensor = nvshmem.interop.torch.empty((1000, 1000), dtype=torch.float32)

    # Create a CuPy array backed by NVSHMEM memory
    array = nvshmem.interop.cupy.empty((1000, 1000), dtype=cp.float32)

These objects can be used with their respective libraries’ APIs just like regular tensors or arrays:

    # Use PyTorch operations
    tensor = tensor + 1.0
    tensor = torch.nn.functional.relu(tensor)

    # Use CuPy operations
    array = array * 2.0
    array = cp.sqrt(array)

### NVSHMEM Operations with PyTorch and CuPy

The same objects can be used directly in NVSHMEM operations:

    import nvshmem.core as nvshmem

    # Get my PE ID and total number of PEs
    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    # Put data to the next PE (PyTorch example)
    next_pe = (my_pe + 1) % n_pes
    nvshmem.put(tensor, tensor, count=tensor.numel(), pe=next_pe)

    # Get data from previous PE (CuPy example)
    prev_pe = (my_pe - 1) % n_pes
    nvshmem.get(array, array, count=array.size, pe=prev_pe)

### Memory Management

These objects are backed by NVSHMEM symmetric memory, so they should be freed when they are no longer needed. If they are not freed, an Exception will be raised when the garbage collector finds them. Unfreed objects are safely freed at program exit.

    # Free the symmetric memory backing these objects
    nvshmem.free_tensor(tensor)
    nvshmem.free_array(array)

This ensures proper cleanup of distributed resources across all PEs.

## Interoperability API reference

### Interoperability with PyTorch

The following are interoperability helpers for NVSHMEM4Py memory used in Torch

`nvshmem.core.interop.torch. bytetensor ( shape: Tuple[int], dtype: None = None, release=False, morder='C', except_on_del=True ) → None`

Create a PyTorch tensor from NVSHMEM-allocated memory with the given shape and dtype.

This function allocates raw memory using NVSHMEM and wraps it in a PyTorch tensor without reshaping or changing the dtype view. Useful for low-level manipulation.

Args:

  * shape (tuple or list of int): Shape of the desired tensor.
  * dtype (`str`, `np.dtype`, or `torch.dtype`, optional): Data type of the tensor. Defaults to `"float32"`.
  * release (bool, optional): Do not track this buffer internally to NVSHMEM
    If True, it is the user’s responsibility to hold references to the buffer until free() is called otherwise, deadlocks may occur.

>   * morder (`str`, optional): The memory format to use. `"C"` for C-style, and `"F"` for “Fortran-style
>

Returns:
    torch.Tensor: A raw PyTorch tensor referencing the NVSHMEM-allocated memory.
Raises:
    RuntimeError: If NVSHMEM or PyTorch is not properly initialized or enabled.

`nvshmem.core.interop.torch. tensor ( shape: Tuple[int], dtype: None = None, release=False, morder='C', except_on_del=True ) → None`

Create a PyTorch tensor view on NVSHMEM-allocated memory with the given shape and dtype.

This function allocates memory using NVSHMEM, wraps it with a DLPack tensor, and then converts it into a PyTorch tensor with the desired dtype and shape.

Args:

  * shape (tuple or list of int): Shape of the desired tensor.
  * dtype (torch.dtype, optional): Data type of the tensor. Defaults to `torch.float32`.
  * release (bool, optional): Do not track this buffer internally to NVSHMEM
    If True, it is the user’s responsibility to hold references to the buffer until free() is called otherwise, deadlocks may occur.
  * morder (`str`, optional): The memory format to use. `"C"` for C-style, and `"F"` for “Fortran-style

Returns:
    torch.Tensor: A PyTorch tensor view on the NVSHMEM-allocated buffer.
Raises:
    RuntimeError: If NVSHMEM or PyTorch is not properly initialized or enabled.

`nvshmem.core.interop.torch. free_tensor ( tensor: None ) → None`

Free an NVSHMEM-backed Torch Tensor

Args:
    tensor (`torch.Tensor`): A PyTorch tensor backed by NVSHMEM memory.
Returns:
    `None`
Raises:
    `RuntimeError`: If NVSHMEM or PyTorch is not properly initialized or enabled.

`nvshmem.core.interop.torch. tensor_get_buffer ( tensor: None ) → Tuple[cuda.core._memory._buffer.Buffer, int, str]`

Get a nvshmem Buffer object from a Torch tensor object which was allocated with `nvshmem.core.tensor()` or `nvshmem.core.bytetensor()` Returns the buffer and the array’s size

`nvshmem.core.interop.torch. get_peer_tensor ( tensor: None , peer_pe: int = None ) → None`

Return a Buffer based on the `peer_buffer` (wrapper of nvshmem_ptr) API

`nvshmem.core.interop.torch. get_multicast_tensor ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , tensor: None ) → None`

Returns a PyTorch Tensor view on multicast-accessible memory corresponding to the input tensor.

This function takes a PyTorch tensor that wraps NVSHMEM-allocated memory and returns a new tensor that uses a Multicast Memory alias for that buffer, obtained via `nvshmemx_mc_ptr`. The resulting tensor is suitable for use in GPU kernels that leverage multicast features such as NVSwitch-based SHARP collectives.

The Tensor passed into it must be allocated by NVSHMEM4Py.

IMPORTANT:

  * The returned tensor’s memory cannot be accessed from the host (CPU). It is only valid for use in GPU kernels. Host-side access or copying is undefined behavior and may result in errors.

NOTE: This function does not copy data. It provides a device-side view of the same underlying memory, but aliased for multicast access. Any modifications made through the returned tensor will affect the same memory region.

Args:
    tensor (`torch.Tensor`): A PyTorch tensor backed by NVSHMEM-allocated memory. team (`Teams`): The NVSHMEM team for which multicast access is requested.
Returns:
    `torch.Tensor`: A PyTorch tensor view on the multicast alias of the NVSHMEM buffer.
Raises:
    `NvshmemInvalid`: If the input tensor is not backed by NVSHMEM memory or multicast is not supported. `NvshmemError`: If the tensor’s backing buffer is not properly tracked or initialized.

`nvshmem.core.interop.torch. register_external_tensor ( tensor: None ) → None`

Register an external tensor with NVSHMEM.

`nvshmem.core.interop.torch. unregister_external_tensor ( tensor: None ) → None`

Unregister an external tensor with NVSHMEM.

### Interoperability with CuPy

The following are interoperability helpers for NVSHMEM4Py memory used in CuPy

`nvshmem.core.interop.cupy. bytearray ( shape: Tuple[int], dtype: str = 'float32', release=False, device_id: int = None, morder='C', except_on_del=True ) → None`

Create a raw CuPy byte array from NVSHMEM-allocated memory.

This function allocates raw memory using NVSHMEM and wraps it with a CuPy array without reshaping or reinterpreting the dtype view.

This function uses the shape and dtype to choose how much memory to allocate, but does not cast or reshape Therefore, the type of the array will always be cupy.uint8.

Any future calls to `.view()` on this object should set `copy=False`, to avoid copying the object off of the sheap

Args:

  * shape (tuple or list of int): Shape of the desired array.
  * dtype (`str`, `np.dtype`, or `cupy.dtype`, optional): Data type of the array. Defaults to `"float32"`.
  * release (bool, optional): Do not track this buffer internally to NVSHMEM
    If True, it is the user’s responsibility to hold references to the buffer until free() is called otherwise, deadlocks may occur.
  * morder (`str`, optional): The memory format to use. `"C"` for C-style, and `"F"` for “Fortran-style

Returns:
    `cupy.ndarray`: A CuPy array backed by NVSHMEM-allocated memory.
Raises:
    `ModuleNotFoundError`: If CuPy is not available or enabled.

`nvshmem.core.interop.cupy. array ( shape: Tuple[int], dtype: str = 'float32', release=False, morder='C', except_on_del=True ) → None`

Create a CuPy array view on NVSHMEM-allocated memory with the given shape and dtype.

This function allocates memory using NVSHMEM, wraps it with a DLPack-compatible CuPy array, and returns a reshaped and retyped view of that memory.

Args:

  * shape (tuple or list of int): Shape of the desired array.
  * dtype (`str`, `np.dtype`, or `cupy.dtype`, optional): Data type of the array. Defaults to `"float32"`.
  * release (bool, optional): Do not track this buffer internally to NVSHMEM
    If True, it is the user’s responsibility to hold references to the buffer until free() is called otherwise, deadlocks may occur.
  * morder (`str`, optional): The memory format to use. `"C"` for C-style, and `"F"` for “Fortran-style

Any future calls to `.view()` on this object should set copy=False, to avoid copying the object off of the sheap

Returns:
    `cupy.ndarray`: A CuPy array view on NVSHMEM-allocated memory.
Raises:
    `ModuleNotFoundError`: If CuPy is not available or enabled.

`nvshmem.core.interop.cupy. free_array ( array: None ) → None`

Free an NVSHMEM-backed CuPy Array

Args:
    array (`cupy.ndarray`): A CuPy array backed by NVSHMEM memory.
Returns:
    None
Raises:
    `ModuleNotFoundError`: If CuPy is not available or enabled.

`nvshmem.core.interop.cupy. array_get_buffer ( array: None ) → Tuple[cuda.core._memory._buffer.Buffer, int, str]`

Get a nvshmem Buffer object from a Cupy NDArray object which was allocated with `nvshmem.core.array()` or `nvshmem.core.bytearray()`

Returns a Tuple of the array and its size in bytes

`nvshmem.core.interop.cupy. get_peer_array ( array: None , peer_pe: int = None ) → None`

Return a Buffer based on the peer_buffer (wrapper of nvshmem_ptr) API

`nvshmem.core.interop.cupy. get_multicast_array ( team: <MutableEnum TEAM_INVALID=-1 , TEAM_WORLD=0 , TEAM_SHARED=1 , TEAM_NODE=2 , TEAM_SAME_MYPE_NODE=3 , TEAM_SAME_GPU=4 , TEAM_GPU_LEADERS=5 , TEAM_MC_SHARED=6 , TEAMS_MIN=7 , TEAM_INDEX_MAX=32767> , array: None ) → None`

Returns a CuPy array view on multicast-accessible memory corresponding to the input array.

This function takes a CuPy Array that wraps NVSHMEM-allocated memory and returns a new array that uses a Multicast Memory alias for that buffer, obtained via `nvshmemx_mc_ptr`. The resulting array is suitable for use in GPU kernels that leverage multicast features such as NVSwitch-based SHARP collectives.

The Array passed into it must be allocated by NVSHMEM4Py.

IMPORTANT:

  * The returned array’s memory cannot be accessed from the host (CPU). It is only valid for use in GPU kernels. Host-side access or copying is undefined behavior and may result in errors.

NOTE: This function does not copy data. It provides a device-side view of the same underlying memory, but aliased for multicast access. Any modifications made through the returned array will affect the same memory region.

Args:
    array (`ndarray`): A CuPy Array backed by NVSHMEM-allocated memory. team (`Teams`): The NVSHMEM team for which multicast access is requested.
Returns:
    `ndarray`: A PyTorch array view on the multicast alias of the NVSHMEM buffer.
Raises:
    `NvshmemInvalid`: If the input array is not backed by NVSHMEM memory or multicast is not supported. `NvshmemError`: If the array’s backing buffer is not properly tracked or initialized.

`nvshmem.core.interop.cupy. register_external_array ( array: None ) → None`

Register an external array with NVSHMEM.

`nvshmem.core.interop.cupy. unregister_external_array ( array: None ) → None`

Unregister an external array with NVSHMEM.
